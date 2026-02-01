use std::{
    error::Error,
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

use crate::midievol::{self, Melody, ModFuncParamType, Score, ScoreInfo, Voices};

pub enum TUIEvent {
    Reset,
    NewBPM(f64),
    SetChildren(u32),
    SetXGens(u32),
    SendStart,
    SendStop,
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

fn melody_summary(m: &midievol::Melody, max_notes: usize, cfg: Arc<Mutex<midievol::MidievolConfig>>) -> String {
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
            // keep modal open for now; caller decides to close on success
            ModalOutcome::Submit(s)
        }
        KeyCode::Backspace => {
            modal.input.pop();
            modal.error = None;
            ModalOutcome::Consumed
        }
        KeyCode::Char(c) => {
            if !c.is_control() {
                modal.input.push(c);
                modal.error = None;
            }
            ModalOutcome::Consumed
        }
        _ => ModalOutcome::Consumed,
    }
}

#[derive(Clone, Debug)]
pub enum ModalKind {
    None,
    SetBpm,
    SetChildren,
    SetXGens,
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
        self.error = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.input.clear();
        self.error = None;
    }
}

pub struct App {
    cfg: Arc<Mutex<midievol::MidievolConfig>>,
    melody: Arc<Mutex<Option<midievol::Melody>>>,
    logs: std::collections::VecDeque<String>,
    modal: InputModal,
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
    let cursor_x = area.x + 2 + 2 + modal.input.len() as u16;
    let cursor_y = area.y + 1 + 3;
    f.set_cursor_position((cursor_x - 1, cursor_y));
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
                        format!("{}={:.3} [{},{}] {:?}", p.name, p.value, p.range[0], p.range[1], p.t)
                    }
                    ModFuncParamType::Float => {
                        format!("{}={:.3} [{},{}] {:?}", p.name, p.value, p.range[0], p.range[1], p.t)
                    }
                    ModFuncParamType::Int => {
                        format!("{}={:.0} [{},{}] {:?}", p.name, p.value, p.range[0], p.range[1], p.t)
                    }
                })
                .collect::<Vec<_>>()
                .join(" | ")
        };
    
        let score = scores[idx].clone().unwrap_or(Score { score: 0.0, info: vec![] });
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

    // Each entry is one line (no wrapping awareness, but good enough for most logs)
    let lines: Vec<Line> = app.logs.iter().cloned().map(Line::from).collect();

    let total_lines = lines.len() as u16;

    // Visible height inside the bordered block (minus top+bottom borders)
    let inner_h = area.height.saturating_sub(2);

    // Scroll so the last lines are visible
    let scroll_y = total_lines.saturating_sub(inner_h);

    let logs = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).title("Log"))
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));

    f.render_widget(logs, area);

    // --- Status bar ---
    let status = Paragraph::new(format!(
        "(q)uit | (r)eset | send (p)lay signal | send (s)top signal | {}{} or (b)pm: {:3.0} | (x)_gens: {} | (c)hildren: {} | producer in-flight: {}",
        BPM_DOWN_KEY,
        BPM_UP_KEY,
        cfg.bpm,
        cfg.x_gens,
        cfg.children,
        if app.in_flight { "YES" } else { "no" },
    ));

    f.render_widget(status, outer[3]);

    draw_input_modal(f, &app.modal);
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
                            }
                        }
                        ModalOutcome::Cancel | ModalOutcome::Consumed => { /* nothing else */ }
                        ModalOutcome::NotOpen => {}
                    }

                    // IMPORTANT: don't let modal keys fall through to normal UI
                    continue;
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
