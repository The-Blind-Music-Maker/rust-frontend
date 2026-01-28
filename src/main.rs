use midir::{Ignore, MidiInput, MidiInputPort};
use midir::{MidiOutput, MidiOutputPort};

use std::collections::HashMap;
use std::error::Error;
use std::io::{Write, stdin, stdout};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

mod midievol;
mod scheduler;
mod tui;

use crate::midievol::{Melody, MidievolConfig};
use crate::scheduler::{LoopData, NoteEvent, Scheduler, TrackData, send_realtime};
use crate::tui::{TUIEvent, UiEvent, pitch_x10_to_midi, run_tui};

const MIDI_START: u8 = 0xFA;
const MIDI_STOP: u8 = 0xFC;

#[derive(Clone, Copy, Debug)]
struct MidiCc {
    ch: u8,
    cc: u8,
    val: u8,
}

#[derive(Clone)]
struct EvolveState {
    last_melody: midievol::Melody,
    // anything else you need: cfg.voices, cfg.modfuncs, rng seed, etc.
    cfg: midievol::MidievolConfig,
}

fn clamp_velocity(v: u64) -> u8 {
    v.min(127) as u8
}

fn round_up_to_multiple(x: u64, multiple: u64) -> u64 {
    if multiple == 0 {
        return x;
    }
    let r = x % multiple;
    if r == 0 { x } else { x + (multiple - r) }
}

fn make_metronome_track(
    loop_len_ticks: u64,
    tpq: u64,
    channel: u8,    // e.g. 9 for GM drums
    velocity: u8,   // e.g. 110
    start_note: u8, // e.g. 37 side stick
    snare: u8,      // e.g. 37 side stick
    kick: u8,       // e.g. 76 hi wood block
) -> TrackData {
    let tpq = tpq.max(1);

    // short click
    let click_len = (tpq / 20).max(1);

    let num_quarters = loop_len_ticks / tpq; // loop_len_ticks is multiple of tpq
    let mut events = Vec::with_capacity(num_quarters as usize);

    for i in 0..num_quarters {
        let start = i * tpq;
        let is_snare_beat = (i % 2) == 1;

        if i == 0 {
            events.push(NoteEvent::note(start, start_note, click_len));
        }

        if is_snare_beat {
            events.push(NoteEvent::note(start, snare, click_len));
        }
        events.push(NoteEvent::note(start, kick, click_len));
    }

    TrackData {
        channel: channel & 0x0F,
        velocity,
        default_note: kick,
        events,
    }
}

fn median_length_to_cc(events: &[NoteEvent], min_ticks: u64, max_ticks: u64) -> Option<u8> {
    if events.is_empty() {
        return None;
    }

    // 1) collect lengths
    let mut lengths: Vec<u64> = events.iter().map(|e| e.length_ticks).collect();

    // 2) sort for median
    lengths.sort_unstable();

    // 3) median
    let n = lengths.len();
    let median = if n % 2 == 1 {
        lengths[n / 2]
    } else {
        (lengths[n / 2 - 1] + lengths[n / 2]) / 2
    };

    // 4) clamp
    let clamped = median.clamp(min_ticks, max_ticks);

    // 5) normalize → MIDI CC
    let normalized = (clamped - min_ticks) as f64 / (max_ticks - min_ticks) as f64;

    Some((normalized * 127.0).round() as u8)
}

fn melody_to_loop_data(
    m: &midievol::Melody,
    tpq: u64,
    voices: &midievol::Voices,
    // These are 0-based MIDI channels: 0=ch1, 1=ch2, 2=ch3
    ch_low: u8,
    ch_mid: u8,
    ch_high: u8,
    default_note: u8,
) -> LoopData {
    // Melody track velocity: average volume, clamped 0..127
    let melody_vel: u8 = if m.notes.is_empty() {
        100
    } else {
        let sum: u64 = m.notes.iter().map(|n| n.volume as u64).sum();
        clamp_velocity(sum / (m.notes.len() as u64))
    };

    // Bucket events by pitch thresholds
    let mut low_events: Vec<NoteEvent> = Vec::new();
    let mut mid_events: Vec<NoteEvent> = Vec::new();
    let mut high_events: Vec<NoteEvent> = Vec::new();

    for n in &m.notes {
        let midi_note = pitch_x10_to_midi(n.pitch);

        let ev = NoteEvent::note(n.position as u64, midi_note, (n.length as u64).max(1));

        if midi_note < (voices.min & 0x7F) {
            low_events.push(ev);
        } else if midi_note < (voices.max & 0x7F) {
            mid_events.push(ev);
        } else {
            high_events.push(ev);
        }
    }

    low_events.sort_by_key(|e| e.start_tick);
    let low_median_cc = { median_length_to_cc(&low_events, 75, 2400).unwrap_or(0) };
    mid_events.sort_by_key(|e| e.start_tick);
    let mid_median_cc = { median_length_to_cc(&mid_events, 75, 2400).unwrap_or(0) };
    high_events.sort_by_key(|e| e.start_tick);
    let high_median_cc = { median_length_to_cc(&high_events, 75, 2400).unwrap_or(0) };

    // Compute loop length from all events (across all buckets)
    let raw_end = low_events
        .iter()
        .chain(mid_events.iter())
        .chain(high_events.iter())
        .map(|e| e.start_tick + e.length_ticks)
        .max()
        .unwrap_or(0);

    let loop_len_ticks = round_up_to_multiple(raw_end.max(1), tpq.max(1));

    let mut tracks: Vec<TrackData> = Vec::new();

    // Only add tracks that actually have events (optional, but avoids empty tracks)
    if !low_events.is_empty() {
        tracks.push(TrackData {
            channel: ch_low & 0x0F,
            velocity: melody_vel,
            default_note,
            events: low_events,
        });
    }
    if !mid_events.is_empty() {
        tracks.push(TrackData {
            channel: ch_mid & 0x0F,
            velocity: melody_vel,
            default_note,
            events: mid_events,
        });
    }
    if !high_events.is_empty() {
        tracks.push(TrackData {
            channel: ch_high & 0x0F,
            velocity: melody_vel,
            default_note,
            events: high_events,
        });
    }

    let mut loop_data = LoopData {
        loop_len_ticks,
        tracks,
        voice_mediants: [low_median_cc, mid_median_cc, high_median_cc],
    };

    // Add metronome track (unchanged)
    let met_track = make_metronome_track(
        loop_data.loop_len_ticks,
        tpq,
        9, // GM drums channel (0-based 9 == MIDI channel 10)
        110,
        55,
        51,
        48,
    );

    loop_data.tracks.push(met_track);
    loop_data
}

// ======================= main =======================

fn main() {
    env_logger::init();
    let cfg = midievol::load_config("./config/config.yaml").unwrap();

    let melody = generate_new_melody(&cfg);

    if let Err(e) = run(melody, cfg) {
        eprintln!("Error: {e}");
    }
}

fn run(initial: Melody, cfg: MidievolConfig) -> Result<(), Box<dyn Error>> {
    let mut conn_out = open_midi_output()?;

    let midi_in_data: Option<(MidiInput, MidiInputPort)> = match open_midi_input() {
        Ok(x) => Some(x),
        Err(e) => {
            eprintln!("MIDI input open failed: {e}");
            None
        }
    };

    // ---- Tempo / internal scheduler resolution ----
    let bpm: f64 = 120.0;
    let tpq: u64 = 600; // internal ticks per quarter note (NOT MIDI clock)
    let tick_period = tick_period_from_bpm_tpq(bpm, tpq);

    // MIDI clock output (24 PPQN)
    let send_midi_clock = true;

    // Transport start/stop (optional)
    let send_transport = true;

    if send_midi_clock && tpq % 24 != 0 {
        return Err("tpq must be divisible by 24 to derive exact MIDI clock ticks".into());
    }
    let ticks_per_midi_clock: u64 = tpq / 24; // 600/24 = 25

    // ---- Producer thread: fetch/generate loop updates in parallel ----
    // let (loop_tx, loop_rx) = mpsc::channel::<LoopData>();
    // thread::spawn(move || producer_example(loop_tx, tpq));

    println!("Playback: {bpm} BPM, tpq={tpq} internal ticks/QN");
    println!(
        "MIDI clock: {} (every {} internal ticks)",
        if send_midi_clock { "ON" } else { "OFF" },
        ticks_per_midi_clock
    );
    println!("Press Enter to stop.\n");

    // Transport start (optional)
    if send_transport {
        send_realtime(&mut conn_out, MIDI_START);
    }

    let (stop_tx, stop_rx) = mpsc::channel::<()>();

    let (loop_tx, loop_rx) = mpsc::channel::<LoopData>();
    let (boundary_tx, boundary_rx) = mpsc::sync_channel::<()>(2);

    let (cc_tx, cc_rx) = mpsc::channel::<MidiCc>();
    if let Some(data) = midi_in_data {
        let _midi_in_thread = spawn_midi_cc_listener(cc_tx, data.0, data.1);
    }

    // NEW: UI channel
    let (ui_tx, ui_rx) = mpsc::channel::<UiEvent>();
    let (tui_tx, tui_rx) = mpsc::channel::<TUIEvent>();

    // NEW: shared cfg snapshot for UI
    let cfg_ui = Arc::new(Mutex::new(cfg.clone()));
    let melody_ui: Arc<Mutex<Option<Melody>>> = Arc::new(Mutex::new(Some(initial.clone())));

    // spawn TUI
    // spawn TUI
    {
        let cfg_ui = Arc::clone(&cfg_ui);
        let melody_ui = Arc::clone(&melody_ui); // NEW
        let stop_tx2 = stop_tx.clone();
        thread::spawn(move || {
            if let Err(e) = run_tui(cfg_ui, melody_ui, ui_rx, stop_tx2, tui_tx) {
                eprintln!("TUI error: {e}");
            }
        });
    }

    let loop_data = melody_to_loop_data(
        &initial,
        600,
        &cfg.voices,
        0, // channel 1
        1, // channel 2
        2, // channel 3
        64,
    );

    let mut scheduler = Scheduler::new(
        loop_data,
        tpq,
        bpm,
        tick_period,
        Instant::now(),
        send_midi_clock,
        ticks_per_midi_clock,
        Some(boundary_tx),
    );

    {
        let ui_tx = ui_tx.clone();
        let cfg_ui = Arc::clone(&cfg_ui);
        let melody_ui = Arc::clone(&melody_ui); // NEW
        thread::spawn(move || {
            producer(
                loop_tx,
                boundary_rx,
                cc_rx,
                tpq,
                EvolveState {
                    last_melody: initial,
                    cfg,
                },
                ui_tx,
                tui_rx,
                cfg_ui,
                melody_ui, // NEW
            );
        });
    }

    // Keep newest pending update only (drop intermediate ones)
    let mut pending: Option<LoopData> = None;

    loop {
        if stop_rx.try_recv().is_ok() {
            break;
        }

        // Drain updates without blocking (parallel fetching)
        while let Ok(new_loop) = loop_rx.try_recv() {
            pending = Some(new_loop);
        }

        // Sleep until next event deadline; then process any events due.
        scheduler.wait_until_next_deadline();

        // Process due events and apply pending swap at boundaries
        scheduler.process_due_events(&mut conn_out, &mut pending);
    }

    println!("\nStopping: sending MIDI STOP + all-notes-off.");
    if send_transport {
        scheduler::send_realtime(&mut conn_out, MIDI_STOP);
    }
    scheduler::all_notes_off(&mut conn_out);

    conn_out.close();
    Ok(())
}

// ======================= producer example =======================

fn generate_new_melody(cfg: &MidievolConfig) -> Melody {
    let mut r = rand::rng();
    let dna = midievol::create_random_melody(8, &mut r);

    let init_payload = midievol::InitPayload {
        dna: dna,
        voices: cfg.voices.clone(),
        modfuncs: cfg.modfuncs.clone(),
    };

    midievol::send_init_req("http://localhost:8080/init", init_payload).unwrap()
}

fn producer(
    loop_tx: mpsc::Sender<LoopData>,
    boundary_rx: mpsc::Receiver<()>,
    cc_rx: mpsc::Receiver<MidiCc>,
    tpq: u64,
    mut state: EvolveState,
    ui_tx: mpsc::Sender<UiEvent>,
    tui_rx: mpsc::Receiver<TUIEvent>, // <-- FIXED
    cfg_ui: Arc<Mutex<midievol::MidievolConfig>>,
    melody_ui: Arc<Mutex<Option<midievol::Melody>>>,
) {
    let (result_tx, result_rx) = mpsc::channel::<(u64, Melody)>();
    let mut in_flight = false;
    let mut token: u64 = 0;

    let cc_map = build_cc_map_from_cfg(&state.cfg);

    loop {
        // ---------------- TUI events (Reset) ----------------
        while let Ok(ev) = tui_rx.try_recv() {
            match ev {
                TUIEvent::Reset => {
                    // Invalidate any in-flight evolve result
                    token = token.wrapping_add(1);
                    in_flight = true; // we are now "in flight" for the init request
                    let my_token = token;

                    let _ = ui_tx.send(UiEvent::ProducerInFlight(true));
                    let _ = ui_tx.send(UiEvent::ProducerResult(
                        "reset: generating new melody...".into(),
                    ));

                    // Grab current config to build init payload
                    // (Use cfg_ui snapshot or state.cfg; both should be same-ish)
                    let cfg = { cfg_ui.lock().unwrap().clone() };

                    let tx = result_tx.clone();
                    thread::spawn(move || {
                        let melody = generate_new_melody(&cfg);
                        let _ = tx.send((my_token, melody));
                    });
                }
            }
        }

        // ---------------- CC updates ----------------
        while let Ok(msg) = cc_rx.try_recv() {
            if let Some((target, new_value)) =
                apply_cc_with_map_and_describe(&mut state.cfg, &cc_map, msg)
            {
                *cfg_ui.lock().unwrap() = state.cfg.clone();
                let _ = ui_tx.send(UiEvent::CcApplied {
                    ch: msg.ch,
                    cc: msg.cc,
                    val: msg.val,
                    target,
                    new_value,
                });
            }
        }

        // ---------------- Apply results (init/evolve) ----------------
        while let Ok((res_token, new_melody)) = result_rx.try_recv() {
            // Ignore stale results (e.g., an evolve response arriving after Reset)
            if res_token != token {
                continue;
            }

            in_flight = false;
            let _ = ui_tx.send(UiEvent::ProducerInFlight(false));

            state.last_melody = new_melody;
            *melody_ui.lock().unwrap() = Some(state.last_melody.clone());

            let loop_data =
                melody_to_loop_data(&state.last_melody, tpq, &state.cfg.voices, 0, 1, 2, 64);

            if loop_tx.send(loop_data).is_err() {
                return;
            }

            let _ = ui_tx.send(UiEvent::ProducerResult("new melody applied".into()));
        }

        // ---------------- Boundary tick -> evolve ----------------
        match boundary_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(()) => {
                if !in_flight {
                    in_flight = true;
                    token = token.wrapping_add(1);
                    let my_token = token;

                    let _ = ui_tx.send(UiEvent::ProducerInFlight(true));

                    let tx = result_tx.clone();
                    let payload = midievol::EvolvePayload {
                        dna: state.last_melody.dna.clone(),
                        x_gens: state.cfg.x_gens,
                        children: state.cfg.children,
                        voices: state.cfg.voices.clone(),
                        modfuncs: state.cfg.modfuncs.clone(),
                    };

                    thread::spawn(move || {
                        let melody =
                            midievol::send_evolve_req("http://localhost:8080/evolve", payload)
                                .unwrap();
                        let _ = tx.send((my_token, melody));
                    });
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

// ======================= timing helpers =======================

fn tick_period_from_bpm_tpq(bpm: f64, tpq: u64) -> Duration {
    let secs = 60.0 / (bpm * tpq as f64);
    Duration::from_secs_f64(secs)
}

// ======================= MIDI output selection =======================

fn open_midi_output() -> Result<midir::MidiOutputConnection, Box<dyn Error>> {
    let midi_out = MidiOutput::new("midir-event-scheduler")?;

    let out_ports = midi_out.ports();
    let out_port: &MidiOutputPort = match out_ports.len() {
        0 => return Err("no output port found".into()),
        1 => {
            println!(
                "Choosing the only available output port: {}",
                midi_out.port_name(&out_ports[0]).unwrap()
            );
            &out_ports[0]
        }
        _ => {
            println!("\nAvailable output ports:");
            for (i, p) in out_ports.iter().enumerate() {
                println!("{}: {}", i, midi_out.port_name(p).unwrap());
            }
            print!("Please select output port: ");
            stdout().flush()?;
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            out_ports
                .get(input.trim().parse::<usize>()?)
                .ok_or("invalid output port selected")?
        }
    };

    println!("\nOpening connection");
    let conn_out = midi_out.connect(out_port, "midir-event-scheduler")?;
    println!("Connection open.");
    Ok(conn_out)
}

fn open_midi_input() -> Result<(MidiInput, MidiInputPort), Box<dyn Error>> {
    let mut midi_in = MidiInput::new("midir-cc-listener")?;
    midi_in.ignore(Ignore::None);

    let in_ports = midi_in.ports();
    let in_port: MidiInputPort = match in_ports.len() {
        0 => return Err("no input port found".into()),
        1 => {
            println!(
                "Choosing the only available input port: {}",
                midi_in.port_name(&in_ports[0]).unwrap()
            );
            in_ports[0].clone()
        }
        _ => {
            println!("\nAvailable input ports:");
            for (i, p) in in_ports.iter().enumerate() {
                println!("{}: {}", i, midi_in.port_name(p).unwrap());
            }
            print!("Please select input port: ");
            stdout().flush()?;
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            in_ports
                .get(input.trim().parse::<usize>()?)
                .ok_or("invalid input port selected")?
                .clone()
        }
    };

    Ok((midi_in, in_port))
}

fn spawn_midi_cc_listener(
    cc_tx: mpsc::Sender<MidiCc>,
    midi_in: MidiInput,
    in_port: MidiInputPort,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let conn = midi_in.connect(
            &in_port,
            "midir-cc-listener",
            move |_stamp, message, _| {
                if message.len() == 3 {
                    let status = message[0];
                    if (status & 0xF0) == 0xB0 {
                        let ch = status & 0x0F;
                        let cc = message[1];
                        let val = message[2];
                        // println!("cc: chan: {ch}, cc: {cc}, val: {val}");
                        let _ = cc_tx.send(MidiCc { ch, cc, val });
                    }
                }
            },
            (),
        );

        let _conn = match conn {
            Ok(c) => c,
            Err(e) => {
                eprintln!("MIDI input connect failed: {e}");
                return;
            }
        };

        loop {
            thread::sleep(Duration::from_secs(3600));
        }
    })
}

fn cc_to_unit(val: u8) -> f64 {
    (val as f64) / 127.0
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn map_cc_to_param_value(param: &midievol::ModFuncParam, cc_val: u8) -> f64 {
    let t = cc_to_unit(cc_val);

    let lo = param.range[0] as f64;
    let hi = param.range[1] as f64;

    match param.t {
        midievol::ModFuncParamType::Float => lerp(lo, hi, t),
        midievol::ModFuncParamType::Int | midievol::ModFuncParamType::Note => {
            // integer-ish range; keep as f64 but whole-number
            lerp(lo, hi, t).round()
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CcTarget {
    ModWeight { mod_idx: usize },
    ModParam { mod_idx: usize, param_idx: usize },
}

fn apply_cc_with_map_and_describe(
    cfg: &mut midievol::MidievolConfig,
    map: &std::collections::HashMap<(Option<u8>, u8), CcTarget>,
    msg: MidiCc,
) -> Option<(String, String)> {
    let key_exact = (Some(msg.ch & 0x0F), msg.cc);
    let key_any = (None, msg.cc);

    let target = map.get(&key_exact).or_else(|| map.get(&key_any)).copied()?;

    match target {
        CcTarget::ModWeight { mod_idx } => {
            let mf = cfg.modfuncs.get_mut(mod_idx)?;
            mf.weight = lerp(0.0, 4.0, cc_to_unit(msg.val));
            Some((format!("{}.weight", mf.name), format!("{:.3}", mf.weight)))
        }
        CcTarget::ModParam { mod_idx, param_idx } => {
            let mf = cfg.modfuncs.get_mut(mod_idx)?;
            let p = mf.params.get_mut(param_idx)?;
            p.value = map_cc_to_param_value(p, msg.val);
            Some((format!("{}.{}", mf.name, p.name), format!("{:.3}", p.value)))
        }
    }
}

/// Key = (optional MIDI channel, CC number)
/// - channel: None  => respond on any channel
/// - channel: Some  => respond only on that channel
type CcKey = (Option<u8>, u8);

fn build_cc_map_from_cfg(cfg: &midievol::MidievolConfig) -> HashMap<CcKey, CcTarget> {
    let mut map: HashMap<CcKey, CcTarget> = HashMap::new();

    for (mi, mf) in cfg.modfuncs.iter().enumerate() {
        // ---- weight CC ----
        if let Some(cc) = mf.weight_cc {
            let ch = mf.weight_channel.map(|c| c & 0x0F);
            let key = (ch, cc);

            if map.contains_key(&key) {
                eprintln!(
                    "WARNING: duplicate CC mapping: weight '{}' uses cc={} ch={:?}",
                    mf.name, cc, ch
                );
            }

            map.insert(key, CcTarget::ModWeight { mod_idx: mi });
        }

        // ---- param CCs ----
        for (pi, p) in mf.params.iter().enumerate() {
            if let Some(cc) = p.cc {
                let ch = p.channel.map(|c| c & 0x0F);
                let key = (ch, cc);

                if map.contains_key(&key) {
                    eprintln!(
                        "WARNING: duplicate CC mapping: param '{}::{}' uses cc={} ch={:?}",
                        mf.name, p.name, cc, ch
                    );
                }

                map.insert(
                    key,
                    CcTarget::ModParam {
                        mod_idx: mi,
                        param_idx: pi,
                    },
                );
            }
        }
    }

    map
}
