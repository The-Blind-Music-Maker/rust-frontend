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
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
};

use crate::midievol;

pub enum TUIEvent {
    Reset,
}

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

fn melody_summary(m: &midievol::Melody, max_notes: usize) -> String {
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

    // Basic stats
    parts.push(format!("bpm={}", m.bpm));
    parts.push(format!("score={:.3}", m.score));
    parts.push(format!("notes={}", n));

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
        Cell::from("Params"),
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let rows = cfg.modfuncs.iter().map(|mf| {
        let params = if mf.params.is_empty() {
            "-".to_string()
        } else {
            mf.params
                .iter()
                .map(|p| {
                    format!(
                        "{}={:.3} [{},{}] {:?}",
                        p.name, p.value, p.range[0], p.range[1], p.t
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ")
        };

        Row::new(vec![
            Cell::from(mf.name.clone()),
            Cell::from(format!("{:.3}", mf.weight)),
            Cell::from(params),
        ])
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(26),
            Constraint::Length(8),
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
            melody_summary(m, 40) // reuse your helper
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
        "q: quit | r: reset | producer in-flight: {} | modfuncs: {}",
        if app.in_flight { "YES" } else { "no" },
        cfg.modfuncs.len()
    ));
    f.render_widget(status, outer[3]);
}

#[derive(Clone, Debug)]
pub enum UiEvent {
    // Log(String),
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

pub struct App {
    cfg: Arc<Mutex<midievol::MidievolConfig>>,
    melody: Arc<Mutex<Option<midievol::Melody>>>, // NEW
    logs: std::collections::VecDeque<String>,
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

pub fn run_tui(
    cfg: Arc<Mutex<midievol::MidievolConfig>>,
    melody: Arc<Mutex<Option<midievol::Melody>>>, // NEW
    ui_rx: mpsc::Receiver<UiEvent>,
    stop_tx: mpsc::Sender<()>,
    tui_events_tx: mpsc::Sender<TUIEvent>,
) -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = App::new(cfg, melody);

    // UI tick so the screen refreshes even if no events arrive
    let tick_rate = Duration::from_millis(60);
    let mut last_tick = Instant::now();

    loop {
        // 1) handle keyboard
        if event::poll(Duration::from_millis(1))? {
            if let CEvent::Key(k) = event::read()? {
                if k.code == KeyCode::Char('q') {
                    let _ = stop_tx.send(());
                    break;
                }

                if k.code == KeyCode::Char('r') {
                    let _ = tui_events_tx.send(TUIEvent::Reset);
                }
            }
        }

        // 2) drain UI events
        while let Ok(ev) = ui_rx.try_recv() {
            match ev {
                // UiEvent::Log(s) => app.push_log(s),
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
