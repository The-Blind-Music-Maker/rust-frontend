use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event as CEvent, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, Wrap},
};

use crate::midievol::{self, Melody, MidievolConfig, ModFuncParamType, Score, ScoreInfo, Voices};

pub enum TUIEvent {
    Reset,
    NewBPM(f64),
    SetChildren(u32),
    SetXGens(u32),
    SendStart,
    SendStop,
    LoadConfig(MidievolConfig),
}

const BPM_UP_KEY: &str = "↑";
const BPM_DOWN_KEY: &str = "↓";
const BPM_STEP: f64 = 1.0;
const BPM_MIN: f64 = 20.0;
const BPM_MAX: f64 = 300.0;

const NOTES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

pub fn pitch_x10_to_midi(pitch_x10: u32) -> u8 {
    // Use f64 to round correctly
    let midi = ((pitch_x10 + 240) as f64 / 10.0).round();
    midi.clamp(0.0, 127.0) as u8
}

fn pitch_to_note(pitch: u32) -> &'static str {
    let midi = pitch_x10_to_midi(pitch) as usize;
    NOTES[midi % 12]
}

fn melody_summary(
    m: &midievol::Melody,
    max_notes: usize,
    cfg: Arc<Mutex<midievol::MidievolConfig>>,
) -> String {
    let n = m.notes.len();
    let mut parts: Vec<String> = Vec::new();

    let len_ticks = m
        .notes
        .iter()
        .map(|x| (x.position + x.length) as u64)
        .max()
        .unwrap_or(0);

    // 600 ticks = 1 quarter note, round
    let len_q = (len_ticks + 600 - 1) / 600;

    let notes_per_voice = get_num_notes_per_voice(m, cfg.lock().unwrap().voices.clone());

    // Basic stats
    parts.push(format!("score={:.3}", m.score));
    parts.push(format!("notes={}", n));
    parts.push(format!("notes_per_voice={:?}", notes_per_voice));

    parts.push(format!("len_q={}", len_q));

    // // DNA (truncate so it doesn't blow up the log pane)
    // let dna = if m.dna.len() > 24 {
    //     format!("{}…", &m.dna[..24])
    // } else {
    //     m.dna.clone()
    // };
    // parts.push(format!("dna={}", dna));

    // Preview first few notes as (pos,pitch,len,vol)
    if n > 0 {
        let preview = m
            .notes
            .iter()
            .take(max_notes)
            .map(|x| {
                format!(
                    "({},{}[{}],{},{})",
                    x.position,
                    pitch_x10_to_midi(x.pitch),
                    pitch_to_note(x.pitch),
                    x.length,
                    x.volume
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        if n > max_notes {
            parts.push(format!("preview: {} …", preview));
        } else {
            parts.push(format!("preview: {}", preview));
        }
    }

    parts.join(" | ")
}

#[derive(Clone, Debug)]
pub enum UiEvent {
    Log(String),
    CcApplied {
        ch: u8,
        cc: u8,
        val: u8,
        target: String,
        new_value: String,
    },
    ProducerInFlight(bool),
    ProducerResult(String),
    // Tick,
}

#[derive(Clone, Debug)]
pub struct SelectModal {
    pub open: bool,
    pub title: String,
    pub help: String,
    pub items: Vec<String>,
    pub selected: usize,
    pub error: Option<String>,
}

impl SelectModal {
    pub fn closed() -> Self {
        Self {
            open: false,
            title: String::new(),
            help: String::new(),
            items: vec![],
            selected: 0,
            error: None,
        }
    }

    pub fn open_with(
        &mut self,
        title: impl Into<String>,
        help: impl Into<String>,
        items: Vec<String>,
    ) {
        self.open = true;
        self.title = title.into();
        self.help = help.into();
        self.items = items;
        self.selected = 0;
        self.error = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.items.clear();
        self.selected = 0;
        self.error = None;
    }
}

enum SelectOutcome {
    NotOpen,
    Consumed,
    Submit(String),
    Cancel,
}

fn handle_select_key(modal: &mut SelectModal, code: KeyCode) -> SelectOutcome {
    if !modal.open {
        return SelectOutcome::NotOpen;
    }

    match code {
        KeyCode::Esc => {
            modal.close();
            SelectOutcome::Cancel
        }
        KeyCode::Enter => {
            if modal.items.is_empty() {
                modal.error = Some("No configs found in ./config".to_string());
                return SelectOutcome::Consumed;
            }
            let item = modal.items[modal.selected].clone();
            SelectOutcome::Submit(item)
        }
        KeyCode::Up => {
            if !modal.items.is_empty() {
                modal.selected = modal.selected.saturating_sub(1);
            }
            SelectOutcome::Consumed
        }
        KeyCode::Down => {
            if !modal.items.is_empty() {
                modal.selected = (modal.selected + 1).min(modal.items.len().saturating_sub(1));
            }
            SelectOutcome::Consumed
        }
        KeyCode::Home => {
            modal.selected = 0;
            SelectOutcome::Consumed
        }
        KeyCode::End => {
            if !modal.items.is_empty() {
                modal.selected = modal.items.len() - 1;
            }
            SelectOutcome::Consumed
        }
        _ => SelectOutcome::Consumed,
    }
}

fn list_config_files() -> Vec<String> {
    let mut out = Vec::new();

    if let Ok(rd) = std::fs::read_dir("./config") {
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.ends_with(".yaml") || name.ends_with(".yml") {
                out.push(name.to_string());
            }
        }
    }

    out.sort();
    out
}

fn load_config_yaml(
    path: &std::path::Path,
) -> Result<midievol::MidievolConfig, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let cfg: midievol::MidievolConfig = serde_yaml::from_slice(&bytes)?;
    Ok(cfg)
}

enum ModalOutcome {
    NotOpen,
    Consumed,
    Submit(String),
    Cancel,
}

fn handle_modal_key(modal: &mut InputModal, code: KeyCode) -> ModalOutcome {
    if !modal.open {
        return ModalOutcome::NotOpen;
    }

    match code {
        KeyCode::Esc => {
            modal.close();
            ModalOutcome::Cancel
        }
        KeyCode::Enter => {
            let s = modal.input.trim().to_string();
            ModalOutcome::Submit(s)
        }

        // Cursor movement
        KeyCode::Left => {
            if modal.cursor > 0 {
                // move left one char (unicode-safe)
                let prev = modal.input[..modal.cursor]
                    .char_indices()
                    .last()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                modal.cursor = prev;
            }
            ModalOutcome::Consumed
        }
        KeyCode::Right => {
            if modal.cursor < modal.input.len() {
                // move right one char (unicode-safe)
                let next = modal.input[modal.cursor..]
                    .char_indices()
                    .nth(1)
                    .map(|(i, _)| modal.cursor + i)
                    .unwrap_or(modal.input.len());
                modal.cursor = next;
            }
            ModalOutcome::Consumed
        }
        KeyCode::Home => {
            modal.cursor = 0;
            ModalOutcome::Consumed
        }
        KeyCode::End => {
            modal.cursor = modal.input.len();
            ModalOutcome::Consumed
        }

        // Editing
        KeyCode::Backspace => {
            backspace_at_cursor(modal);
            modal.error = None;
            ModalOutcome::Consumed
        }
        KeyCode::Delete => {
            delete_at_cursor(modal);
            modal.error = None;
            ModalOutcome::Consumed
        }

        // Insert at cursor
        KeyCode::Char(c) => {
            if !c.is_control() {
                insert_char_at(modal, c);
                modal.error = None;
            }
            ModalOutcome::Consumed
        }

        _ => ModalOutcome::Consumed,
    }
}

fn next_free_config_filename(config_dir: &Path) -> String {
    // We pick the lowest free x in config-x.yaml among existing files.
    // If ./config doesn't exist, default to config-1.yaml.
    let mut used: std::collections::HashSet<u32> = std::collections::HashSet::new();

    if let Ok(rd) = fs::read_dir(config_dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };

            // Match "config-<n>.yaml"
            if let Some(rest) = name.strip_prefix("config-") {
                if let Some(num_part) = rest.strip_suffix(".yaml") {
                    if let Ok(n) = num_part.parse::<u32>() {
                        used.insert(n);
                    }
                }
            }
        }
    }

    let mut x: u32 = 1;
    while used.contains(&x) {
        x += 1;
    }

    format!("config-{}.yaml", x)
}

fn sanitize_filename(input: &str) -> Option<String> {
    let s = input.trim();
    if s.is_empty() {
        return None;
    }

    // Disallow path separators to keep saves inside ./config
    if s.contains('/') || s.contains('\\') {
        return None;
    }

    // Optionally enforce .yaml suffix; you can relax this if you want.
    let s = if s.ends_with(".yaml") {
        s.to_string()
    } else {
        format!("{s}.yaml")
    };

    Some(s)
}

fn save_config_yaml(
    config_dir: &Path,
    filename: &str,
    cfg: &midievol::MidievolConfig,
) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(config_dir)?;

    let path = config_dir.join(filename);

    if path.exists() {
        return Err(format!("File already exists: {}", path.display()).into());
    }

    // Requires MidievolConfig: serde::Serialize
    let yaml = serde_yaml::to_string(cfg)?;
    fs::write(&path, yaml)?;

    Ok(path)
}

#[derive(Clone, Debug)]
pub enum ModalKind {
    None,
    SetBpm,
    SetChildren,
    SetXGens,
    SaveConfig,
    // Add more:
    // SetWeight { modfunc_index: usize },
    // SetParam { modfunc_index: usize, param_index: usize },
}

#[derive(Clone, Debug)]
pub struct InputModal {
    pub open: bool,
    pub kind: ModalKind, // NEW
    pub title: String,
    pub help: String,
    pub input: String,
    pub cursor: usize, // NEW
    pub error: Option<String>,
}

impl InputModal {
    pub fn closed() -> Self {
        Self {
            open: false,
            kind: ModalKind::None,
            title: String::new(),
            help: String::new(),
            input: String::new(),
            cursor: 0, // NEW
            error: None,
        }
    }

    pub fn open_with(
        &mut self,
        kind: ModalKind,
        title: impl Into<String>,
        help: impl Into<String>,
        initial: impl Into<String>,
    ) {
        self.open = true;
        self.kind = kind;
        self.title = title.into();
        self.help = help.into();
        self.input = initial.into();
        self.cursor = self.input.len(); // NEW
        self.error = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.input.clear();
        self.cursor = 0; // NEW
        self.error = None;
    }
}

fn clamp_cursor(modal: &mut InputModal) {
    modal.cursor = modal.cursor.min(modal.input.len());
}

fn insert_char_at(modal: &mut InputModal, c: char) {
    clamp_cursor(modal);
    modal.input.insert(modal.cursor, c);
    modal.cursor += c.len_utf8().min(1); // for ASCII-like typing; ok for most use
    // If you want full unicode correctness, see note below.
}

fn backspace_at_cursor(modal: &mut InputModal) {
    clamp_cursor(modal);
    if modal.cursor == 0 {
        return;
    }
    // Remove the char before cursor (unicode-safe)
    let prev_char_start = modal.input[..modal.cursor]
        .char_indices()
        .last()
        .map(|(i, _)| i)
        .unwrap_or(0);
    modal.input.drain(prev_char_start..modal.cursor);
    modal.cursor = prev_char_start;
}

fn delete_at_cursor(modal: &mut InputModal) {
    clamp_cursor(modal);
    if modal.cursor >= modal.input.len() {
        return;
    }
    // Remove the char at cursor (unicode-safe)
    let next = modal.input[modal.cursor..]
        .char_indices()
        .nth(1)
        .map(|(i, _)| modal.cursor + i)
        .unwrap_or(modal.input.len());
    modal.input.drain(modal.cursor..next);
}

pub struct App {
    cfg: Arc<Mutex<midievol::MidievolConfig>>,
    melody: Arc<Mutex<Option<midievol::Melody>>>,
    logs: std::collections::VecDeque<String>,
    modal: InputModal,
    select_modal: SelectModal, // NEW
    in_flight: bool,
}

impl App {
    pub fn new(
        cfg: Arc<Mutex<midievol::MidievolConfig>>,
        melody: Arc<Mutex<Option<midievol::Melody>>>, // NEW
    ) -> Self {
        Self {
            cfg,
            melody,
            logs: std::collections::VecDeque::with_capacity(500),
            in_flight: false,
            modal: InputModal::closed(),
            select_modal: SelectModal::closed(), // NEW
        }
    }

    pub fn push_log(&mut self, s: impl Into<String>) {
        let s = s.into();
        if self.logs.len() == 500 {
            self.logs.pop_front();
        }
        self.logs.push_back(s);
    }
}

fn get_num_notes_per_voice(m: &Melody, voices: Voices) -> (usize, usize, usize) {
    let mut low_events: usize = 0;
    let mut mid_events: usize = 0;
    let mut high_events: usize = 0;

    for n in &m.notes {
        if n.pitch < (voices.min as u32 * 10) {
            low_events += 1;
        } else if n.pitch < (voices.max as u32 * 10) {
            mid_events += 1;
        } else {
            high_events += 1;
        }
    }

    (low_events, mid_events, high_events)
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn wrap_logs_to_lines(
    logs: &std::collections::VecDeque<String>,
    wrap_width: usize,
) -> Vec<Line<'static>> {
    use textwrap::Options;

    let opts = Options::new(wrap_width)
        .break_words(true) // long tokens still wrap
        .wrap_algorithm(textwrap::WrapAlgorithm::FirstFit);

    let mut out: Vec<Line<'static>> = Vec::new();

    for (i, s) in logs.iter().enumerate() {
        // Preserve empty lines and also handle embedded '\n'
        let parts = s.split('\n');

        for part in parts {
            if part.is_empty() {
                out.push(Line::from(String::new()));
                continue;
            }

            for w in textwrap::wrap(part, &opts) {
                out.push(Line::from(w.into_owned()));
            }
        }

        // Optional: add a blank line between entries
        // if i + 1 < logs.len() { out.push(Line::from("")); }
        let _ = i;
    }

    out
}

fn draw_input_modal(f: &mut Frame, modal: &InputModal) {
    if !modal.open {
        return;
    }

    let area = centered_rect(55, 25, f.area());
    f.render_widget(Clear, area);

    let mut lines = vec![
        Line::from(modal.help.clone()),
        Line::from("Esc cancels • Enter confirms"),
        Line::from(""),
        Line::from(format!("> {}", modal.input)),
    ];

    if let Some(err) = &modal.error {
        lines.push(Line::from(""));
        lines.push(Line::from(err.clone()));
    }

    let p = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(modal.title.clone()),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(p, area);

    // cursor after "> "
    let cursor_x = area.x + 2 + 2 + modal.cursor as u16;
    let cursor_y = area.y + 1 + 3;
    f.set_cursor_position((cursor_x - 1, cursor_y));
}

fn draw_select_modal(f: &mut Frame, modal: &SelectModal) {
    if !modal.open {
        return;
    }

    let area = centered_rect(60, 45, f.area());
    f.render_widget(Clear, area);

    let mut lines: Vec<Line<'static>> = vec![
        Line::from(modal.help.clone()),
        Line::from("Esc cancels • Enter loads • ↑/↓ select • Home/End"),
        Line::from(""),
    ];

    if modal.items.is_empty() {
        lines.push(Line::from("(no .yaml files found in ./config)".to_string()));
    } else {
        // Show up to N items that fit, with a simple window around selected
        let max_visible = (area.height as usize).saturating_sub(6).max(3);
        let sel = modal.selected;

        let start = sel.saturating_sub(max_visible / 2);
        let end = (start + max_visible).min(modal.items.len());
        let start = end.saturating_sub(max_visible).min(start);

        for (i, item) in modal.items[start..end].iter().enumerate() {
            let idx = start + i;
            if idx == sel {
                lines.push(Line::from(format!("> {item}")));
            } else {
                lines.push(Line::from(format!("  {item}")));
            }
        }
    }

    if let Some(err) = &modal.error {
        lines.push(Line::from(""));
        lines.push(Line::from(err.clone()));
    }

    let p = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(modal.title.clone()),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(p, area);
}

fn extract_info(info: Vec<ScoreInfo>) -> String {
    if info.len() > 0 {
        let name = &info[0].name;
        let value = &info[0].value;
        return format!("{name}: {value}");
    }

    return "".into();
}

fn draw_ui(f: &mut Frame, app: &App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),    // config table
            Constraint::Length(7), // NEW: current melody pane
            Constraint::Length(8), // log pane
            Constraint::Length(1), // status bar
        ])
        .split(f.area());

    // --- Config table ---
    let cfg = app.cfg.lock().unwrap().clone();

    let header = Row::new(vec![
        Cell::from("ModFunc"),
        Cell::from("Weight"),
        Cell::from("Score"),
        Cell::from("Info"),
        Cell::from("Params"),
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let scores = {
        let m = app.melody.lock().unwrap();
        if let Some(m) = m.as_ref() {
            m.scores_per_func.clone()
        } else {
            vec![]
        }
    };

    let mut info_len = 2;
    let mut rows = Vec::new();

    for (idx, mf) in cfg.modfuncs.iter().enumerate() {
        let params = if mf.params.is_empty() {
            "-".to_string()
        } else {
            mf.params
                .iter()
                .map(|p| match p.t {
                    ModFuncParamType::Note => {
                        format!(
                            "{}={:.3} [{},{}] {:?}",
                            p.name, p.value, p.range[0], p.range[1], p.t
                        )
                    }
                    ModFuncParamType::Float => {
                        format!(
                            "{}={:.3} [{},{}] {:?}",
                            p.name, p.value, p.range[0], p.range[1], p.t
                        )
                    }
                    ModFuncParamType::Int => {
                        format!(
                            "{}={:.0} [{},{}] {:?}",
                            p.name, p.value, p.range[0], p.range[1], p.t
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join(" | ")
        };

        let score = scores[idx].clone().unwrap_or(Score {
            score: 0.0,
            info: vec![],
        });

        let info = extract_info(score.info);

        info_len = info_len.max(info.len());

        rows.push(Row::new(vec![
            Cell::from(mf.name.clone()),
            Cell::from(format!("{:.1}", mf.weight)),
            Cell::from(format!("{:.3}", score.score)),
            Cell::from(format!("{info}")),
            Cell::from(params),
        ]));
    }

    let table = Table::new(
        rows,
        [
            Constraint::Length(26),
            Constraint::Length(8),
            Constraint::Length(5),
            Constraint::Length(info_len.try_into().unwrap()),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Config (live)"),
    )
    .column_spacing(1);

    f.render_widget(table, outer[0]);

    let melody_text = {
        let m = app.melody.lock().unwrap();
        if let Some(m) = m.as_ref() {
            melody_summary(m, 40, app.cfg.clone()) // reuse your helper
        } else {
            "No melody yet".to_string()
        }
    };

    let melody_box = Paragraph::new(melody_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Current Melody"),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(melody_box, outer[1]);

    // --- Logs (auto-follow newest) ---
    let area = outer[2];

    // inside borders: -2 width and -2 height
    let inner_w = area.width.saturating_sub(2) as usize;
    let inner_h = area.height.saturating_sub(2) as u16;

    // Avoid division by zero / weird widths
    let inner_w = inner_w.max(1);

    // Build wrapped visual lines
    let wrapped_lines: Vec<Line> = wrap_logs_to_lines(&app.logs, inner_w);

    let total_visual_lines = wrapped_lines.len() as u16;

    // Scroll so the last visual lines are visible
    let scroll_y = total_visual_lines.saturating_sub(inner_h);

    let logs = Paragraph::new(Text::from(wrapped_lines))
        .block(Block::default().borders(Borders::ALL).title("Log"))
        // IMPORTANT: we already wrapped manually, so don't wrap again
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));

    f.render_widget(logs, area);

    // --- Status bar ---
    let status = Paragraph::new(format!(
        "(q)uit | (r)eset | (w)rite config | (l)oad config | send (p)lay signal | send (s)top signal | {}{} or (b)pm: {:3.0} | (x)_gens: {} | (c)hildren: {} | producer in-flight: {}",
        BPM_DOWN_KEY,
        BPM_UP_KEY,
        cfg.bpm,
        cfg.x_gens,
        cfg.children,
        if app.in_flight { "YES" } else { "no" },
    ));

    f.render_widget(status, outer[3]);

    draw_input_modal(f, &app.modal);
    draw_select_modal(f, &app.select_modal);
}

pub fn run_tui(
    cfg: Arc<Mutex<midievol::MidievolConfig>>,
    melody: Arc<Mutex<Option<midievol::Melody>>>, // NEW
    ui_rx: mpsc::Receiver<UiEvent>,
    stop_tx: mpsc::Sender<()>,
    tui_events_tx: mpsc::Sender<TUIEvent>,
    tui_scheduler_tx: mpsc::Sender<TUIEvent>,
) -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;
    let mut app = App::new(Arc::clone(&cfg), melody);

    // UI tick so the screen refreshes even if no events arrive
    let tick_rate = Duration::from_millis(60);
    let mut last_tick = Instant::now();

    loop {
        // 1) handle keyboard
        if event::poll(Duration::from_millis(1))? {
            if let CEvent::Key(k) = event::read()? {
                // 1) If modal is open, route everything to it and stop.
                // If modal is open, it consumes input first
                if app.modal.open {
                    let kind = app.modal.kind.clone(); // copy kind now (no borrow issues)
                    let outcome = handle_modal_key(&mut app.modal, k.code);

                    match outcome {
                        ModalOutcome::Submit(txt) => {
                            // Apply based on kind (NOW it's safe to touch app/cfg/send)
                            match kind {
                                ModalKind::SetBpm => {
                                    match txt.parse::<f64>() {
                                        Ok(mut bpm) => {
                                            bpm = bpm.clamp(BPM_MIN, BPM_MAX);
                                            {
                                                let mut cfg = cfg.lock().unwrap();
                                                cfg.bpm = bpm;
                                            }
                                            let _ = tui_scheduler_tx.send(TUIEvent::NewBPM(bpm));
                                            app.push_log(format!("[ui] bpm set to {bpm:.1}"));
                                            app.modal.close();
                                        }
                                        Err(_) => {
                                            app.modal.error = Some("Invalid number".to_string());
                                            // keep modal open
                                        }
                                    }
                                }
                                ModalKind::SetChildren => {
                                    match txt.parse::<u32>() {
                                        Ok(mut children) => {
                                            children = children.clamp(1, u32::max_value());
                                            {
                                                let mut cfg = cfg.lock().unwrap();
                                                cfg.children = children;
                                            }
                                            let _ =
                                                tui_events_tx.send(TUIEvent::SetChildren(children));
                                            app.push_log(format!(
                                                "[ui] children set to {children}"
                                            ));
                                            app.modal.close();
                                        }
                                        Err(_) => {
                                            app.modal.error = Some("Invalid number".to_string());
                                            // keep modal open
                                        }
                                    }
                                }
                                ModalKind::SetXGens => {
                                    match txt.parse::<u32>() {
                                        Ok(mut x_gens) => {
                                            x_gens = x_gens.clamp(1, u32::max_value());
                                            {
                                                let mut cfg = cfg.lock().unwrap();
                                                cfg.x_gens = x_gens;
                                            }
                                            let _ = tui_events_tx.send(TUIEvent::SetXGens(x_gens));
                                            app.push_log(format!("[ui] x_gens set to {x_gens}"));
                                            app.modal.close();
                                        }
                                        Err(_) => {
                                            app.modal.error = Some("Invalid number".to_string());
                                            // keep modal open
                                        }
                                    }
                                }
                                ModalKind::None => {
                                    app.modal.close();
                                } // add more modal kinds here later
                                ModalKind::SaveConfig => {
                                    let Some(fname) = sanitize_filename(&txt) else {
                                        app.modal.error = Some(
                                            "Invalid filename (no slashes; not empty).".to_string(),
                                        );
                                        // keep modal open
                                        continue;
                                    };

                                    // Snapshot current cfg
                                    let cfg_snapshot = cfg.lock().unwrap().clone();

                                    match save_config_yaml(
                                        Path::new("./config"),
                                        &fname,
                                        &cfg_snapshot,
                                    ) {
                                        Ok(path) => {
                                            app.push_log(format!(
                                                "[ui] saved config to {}",
                                                path.display()
                                            ));
                                            app.modal.close();
                                        }
                                        Err(e) => {
                                            app.modal.error = Some(format!("{e}"));
                                            // keep modal open
                                        }
                                    }
                                }
                            }
                        }
                        ModalOutcome::Cancel | ModalOutcome::Consumed => { /* nothing else */ }
                        ModalOutcome::NotOpen => {}
                    }

                    // IMPORTANT: don't let modal keys fall through to normal UI
                    continue;
                }

                // If select modal is open, it consumes input first
                if app.select_modal.open {
                    let outcome = handle_select_key(&mut app.select_modal, k.code);

                    match outcome {
                        SelectOutcome::Submit(filename) => {
                            let path = std::path::Path::new("./config").join(&filename);

                            match load_config_yaml(&path) {
                                Ok(new_cfg) => {
                                    let _ = tui_events_tx.send(TUIEvent::LoadConfig(new_cfg));
                                    app.push_log(format!("[ui] loaded config: {}", path.display()));
                                    app.select_modal.close();
                                }
                                Err(e) => {
                                    // Keep the modal open and show error
                                    app.select_modal.error =
                                        Some(format!("Failed to load {}: {e}", path.display()));
                                }
                            }
                        }

                        SelectOutcome::Cancel | SelectOutcome::Consumed => {}
                        SelectOutcome::NotOpen => {}
                    }

                    continue; // don't fall through
                }

                // 2) Normal keys (open modal etc)
                if k.code == KeyCode::Char('b') {
                    let current = cfg.lock().unwrap().bpm;
                    app.modal.open_with(
                        ModalKind::SetBpm,
                        "Set BPM",
                        "Type a BPM (20..300) and press Enter.",
                        format!("{current:.0}"),
                    );
                    continue;
                }

                if k.code == KeyCode::Char('x') {
                    let current = cfg.lock().unwrap().x_gens;
                    app.modal.open_with(
                        ModalKind::SetXGens,
                        "Set X-Gens",
                        "Type a X-Gens (1..inf) and press Enter.",
                        format!("{current:.0}"),
                    );
                    continue;
                }

                if k.code == KeyCode::Char('c') {
                    let current = cfg.lock().unwrap().children;
                    app.modal.open_with(
                        ModalKind::SetChildren,
                        "Set Children",
                        "Type a Children (1..inf) and press Enter.",
                        format!("{current:.0}"),
                    );
                    continue;
                }

                if k.code == KeyCode::Char('q') {
                    let _ = stop_tx.send(());
                    break;
                }

                if k.code == KeyCode::Char('r') {
                    let _ = tui_events_tx.send(TUIEvent::Reset);
                }

                if k.code == KeyCode::Char('p') {
                    let _ = tui_scheduler_tx.send(TUIEvent::SendStart);
                }

                if k.code == KeyCode::Char('s') {
                    let _ = tui_scheduler_tx.send(TUIEvent::SendStop);
                }

                if k.code == KeyCode::Up {
                    let new_bpm = {
                        let mut cfg = cfg.lock().unwrap();
                        cfg.bpm = (cfg.bpm + BPM_STEP).clamp(BPM_MIN, BPM_MAX);
                        cfg.bpm
                    };
                    let _ = tui_scheduler_tx.send(TUIEvent::NewBPM(new_bpm));
                }

                if k.code == KeyCode::Down {
                    let new_bpm = {
                        let mut cfg = cfg.lock().unwrap();
                        cfg.bpm = (cfg.bpm - BPM_STEP).clamp(BPM_MIN, BPM_MAX);
                        cfg.bpm
                    };
                    let _ = tui_scheduler_tx.send(TUIEvent::NewBPM(new_bpm));
                }

                if k.code == KeyCode::Char('w') {
                    let config_dir = Path::new("./config");
                    let default_name = next_free_config_filename(config_dir);

                    app.modal.open_with(
                        ModalKind::SaveConfig,
                        "Save Config",
                        "Type a filename (saved into ./config).",
                        default_name,
                    );
                    continue;
                }

                if k.code == KeyCode::Char('l') {
                    let items = list_config_files();
                    app.select_modal.open_with(
                        "Load Config",
                        "Select a config from ./config",
                        items,
                    );
                    continue;
                }
            }
        }

        // 2) drain UI events
        while let Ok(ev) = ui_rx.try_recv() {
            match ev {
                UiEvent::Log(s) => app.push_log(s),
                UiEvent::ProducerInFlight(x) => app.in_flight = x,
                UiEvent::ProducerResult(s) => app.push_log(format!("[producer] {s}")),
                UiEvent::CcApplied {
                    ch,
                    cc,
                    val,
                    target,
                    new_value,
                } => {
                    app.push_log(format!(
                        "[cc] ch={ch} cc={cc} val={val} -> {target}={new_value}"
                    ));
                } // UiEvent::Tick => {}
            }
        }

        // 3) periodic tick
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }

        // 4) draw
        terminal.draw(|f| draw_ui(f, &app))?;
    }

    // teardown
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
