use midir::{MidiOutput, MidiOutputPort};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::io::{Write, stdin, stdout};
use std::sync::mpsc;
use std::sync::mpsc::SyncSender;
use std::thread;
use std::time::{Duration, Instant};

use crate::midievol::{Melody, MidievolConfig};

mod midievol;

const MIDI_CLOCK: u8 = 0xF8;
const MIDI_START: u8 = 0xFA;
const MIDI_STOP: u8 = 0xFC;

#[derive(Clone)]
struct EvolveState {
    last_melody: midievol::Melody,
    // anything else you need: cfg.voices, cfg.modfuncs, rng seed, etc.
    cfg: midievol::MidievolConfig,
}

fn clamp_velocity(v: u64) -> u8 {
    v.min(127) as u8
}

fn pitch_x10_to_midi(pitch_x10: u32) -> u8 {
    // Use f64 to round correctly
    let midi = ((pitch_x10 + 240) as f64 / 10.0).round();
    midi.clamp(0.0, 127.0) as u8
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

fn melody_to_loop_data(
    m: &midievol::Melody,
    tpq: u64,
    melody_channel: u8,
    default_note: u8,
) -> LoopData {
    // Melody track velocity: average volume, clamped 0..127
    let melody_vel: u8 = if m.notes.is_empty() {
        100
    } else {
        let sum: u64 = m.notes.iter().map(|n| n.volume as u64).sum();
        clamp_velocity(sum / (m.notes.len() as u64))
    };

    // Melody events
    let mut melody_events: Vec<NoteEvent> = m
        .notes
        .iter()
        .map(|n| {
            let midi_note = pitch_x10_to_midi(n.pitch);
            NoteEvent::note(n.position as u64, midi_note, (n.length as u64).max(1))
        })
        .collect();

    melody_events.sort_by_key(|e| e.start_tick);

    // Raw end + round up to whole quarter notes
    let raw_end = melody_events
        .iter()
        .map(|e| e.start_tick + e.length_ticks)
        .max()
        .unwrap_or(0);

    let loop_len_ticks = round_up_to_multiple(raw_end.max(1), tpq.max(1));

    // Build loop data with melody track
    let mut loop_data = LoopData {
        loop_len_ticks,
        tracks: vec![TrackData {
            channel: melody_channel & 0x0F,
            velocity: melody_vel,
            default_note,
            events: melody_events,
        }],
    };

    // Add metronome track
    // Defaults: channel 9 (GM drums), accented downbeat.
    let met_track = make_metronome_track(
        loop_data.loop_len_ticks,
        tpq,
        9,   // GM drum channel (0-based 9 == MIDI channel 10)
        110, // metronome velocity
        55,  // side stick (accent)
        51,  // side stick (accent)
        48,  // hi wood block (regular)
    );

    loop_data.tracks.push(met_track);
    loop_data
}

// ======================= main =======================

fn main() {
    env_logger::init();
    let cfg = midievol::load_config("./config/config.yaml").unwrap();

    for x in &cfg.modfuncs {
        let name = x.name.clone();
        let weight = x.weight;
        println!("Name: {name}: weight: {weight}");
    }

    let mut r = rand::rng();
    let dna = midievol::create_random_melody(8, &mut r);

    let init_payload = midievol::InitPayload {
        dna: dna,
        voices: cfg.voices.clone(),
        modfuncs: cfg.modfuncs.clone(),
    };

    let melody = midievol::send_init_req("http://localhost:8080/init", init_payload).unwrap();

    if let Err(e) = run(melody, cfg) {
        eprintln!("Error: {e}");
    }
}

fn run(initial: Melody, cfg: MidievolConfig) -> Result<(), Box<dyn Error>> {
    let mut conn_out = open_midi_output()?;

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

    let (loop_tx, loop_rx) = mpsc::channel::<LoopData>();
    // boundary triggers: small bounded buffer so scheduler never blocks
    let (boundary_tx, boundary_rx) = mpsc::sync_channel::<()>(2);

    let loop_data = melody_to_loop_data(&initial, 600, 1, 64);

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

    thread::spawn(move || {
        producer_example(
            loop_tx,
            boundary_rx,
            tpq,
            EvolveState {
                last_melody: initial,
                cfg: cfg,
            },
        )
    });

    // Stopper thread
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    thread::spawn(move || {
        let mut s = String::new();
        let _ = stdin().read_line(&mut s);
        let _ = stop_tx.send(());
    });

    // // ---- Event-driven scheduler ----
    // let mut scheduler = Scheduler::new(
    //     initial,
    //     tpq,
    //     bpm,
    //     tick_period,
    //     Instant::now(),
    //     send_midi_clock,
    //     ticks_per_midi_clock,
    // );

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
        send_realtime(&mut conn_out, MIDI_STOP);
    }
    all_notes_off(&mut conn_out);

    conn_out.close();
    Ok(())
}

// ======================= producer example =======================

fn producer_example(
    loop_tx: mpsc::Sender<LoopData>,
    boundary_rx: mpsc::Receiver<()>,
    tpq: u64,
    mut state: EvolveState,
) {
    // channel for fetch thread to send back result
    let (result_tx, result_rx) = mpsc::channel::<Melody>();

    // single in-flight flag
    let mut in_flight = false;

    // demo toggle (remove when API hooked up)
    let mut flip = false;

    loop {
        // Drain results quickly (usually only one)
        // 1) If fetch returned, update state + forward loop
        while let Ok(new_melody) = result_rx.try_recv() {
            in_flight = false;

            // update state for next evolve call
            state.last_melody = new_melody;

            // convert to LoopData + send to playback
            let loop_data = melody_to_loop_data(&state.last_melody, tpq, 1, 64);
            if loop_tx.send(loop_data).is_err() {
                return;
            }
        }

        // 2) Wait for next boundary trigger (blocks)
        // (We also want to wake periodically to notice result_rx; easiest is recv_timeout)
        match boundary_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(()) => {
                // boundary happened: start a fetch if none in flight
                if !in_flight {
                    in_flight = true;

                    let tx = result_tx.clone();

                    // Build payload from *current state* (clone only what you need)
                    let payload = midievol::EvolvePayload {
                        dna: state.last_melody.dna.clone(),
                        x_gens: state.cfg.x_gens,
                        children: state.cfg.children,
                        voices: state.cfg.voices.clone(),
                        modfuncs: state.cfg.modfuncs.clone(),
                    };
                    // IMPORTANT: move any payload/state needed for the API call into this closure
                    thread::spawn(move || {
                        // ---- Replace demo with real API call ----
                        let melody =
                            midievol::send_evolve_req("http://localhost:8080/evolve", payload)
                                .unwrap();
                        let _ = tx.send(melody);
                    });

                    // demo toggle update (ONLY for the demo path)
                    flip = !flip;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // just loop; this allows us to check result_rx periodically
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return; // scheduler gone
            }
        }
    }
}

// fn example_loop_a(tpq: u64) -> LoopData {
//     // 1 bar of 4/4
//     let loop_len_ticks = 6 * tpq;

//     LoopData {
//         loop_len_ticks,
//         tracks: vec![TrackData {
//             channel: 1,
//             velocity: 90,
//             default_note: 60,
//             events: vec![
//                 NoteEvent::note(0, 61, tpq / 2),            // beat 1, 1/8
//                 NoteEvent::note(tpq / 2 + 75, 62, tpq / 4), // off-grid
//                 NoteEvent::note(tpq, 63, tpq / 2),          // beat 2
//                 NoteEvent::note(2 * tpq - 10, 67, tpq / 3), // off-grid
//                 NoteEvent::note(3 * tpq, 70, tpq / 2),      // beat 4
//                 NoteEvent::note(4 * tpq, 72, tpq / 2),      // beat 5
//                 NoteEvent::note(5 * tpq, 74, tpq / 2),      // beat 6
//             ],
//         }],
//     }
// }

// fn example_loop_b(tpq: u64) -> LoopData {
//     let loop_len_ticks = 4 * tpq;

//     LoopData {
//         loop_len_ticks,
//         tracks: vec![TrackData {
//             channel: 1,
//             velocity: 95,
//             default_note: 60,
//             events: vec![
//                 NoteEvent::note(0, 72, tpq / 4),
//                 NoteEvent::note(tpq / 2, 70, tpq / 2),
//                 NoteEvent::note(tpq + tpq / 4, 67, tpq / 4),
//                 NoteEvent::note(2 * tpq + 30, 63, tpq / 2),
//                 NoteEvent::note(3 * tpq + tpq / 2, 62, tpq / 4),
//             ],
//         }],
//     }
// }

// ======================= data model =======================

#[derive(Clone, Debug)]
struct LoopData {
    loop_len_ticks: u64,
    tracks: Vec<TrackData>,
}

#[derive(Clone, Debug)]
struct TrackData {
    channel: u8,
    velocity: u8,
    default_note: u8,
    events: Vec<NoteEvent>,
}

#[derive(Clone, Copy, Debug)]
struct NoteEvent {
    start_tick: u64,   // position within loop [0..loop_len_ticks)
    note: u8,          // if 0, use track default_note
    length_ticks: u64, // duration in internal ticks
}

impl NoteEvent {
    fn note(start_tick: u64, note: u8, length_ticks: u64) -> Self {
        Self {
            start_tick,
            note,
            length_ticks,
        }
    }
}

// ======================= scheduler =======================

#[derive(Clone, Copy, Debug)]
enum EventKind {
    NoteOn { ch: u8, note: u8, vel: u8 },
    NoteOff { ch: u8, note: u8 },
    Boundary,  // used to apply pending pattern swaps cleanly
    MidiClock, // MIDI realtime clock (0xF8)
}

#[derive(Clone, Copy, Debug)]
struct ScheduledEvent {
    abs_tick: u64,
    kind: EventKind,
    period: u64, // 0 => one-shot; otherwise reschedule by adding period
}

// Lower number = higher priority (processed first) when abs_tick ties
fn kind_priority(k: &EventKind) -> u8 {
    match k {
        EventKind::Boundary => 0, // apply swap before anything else at the boundary
        EventKind::NoteOff { .. } => 1, // turn off before on
        EventKind::NoteOn { .. } => 2,
        EventKind::MidiClock => 3, // clock can be after note events if they coincide
    }
}

// BinaryHeap is a max-heap, so reverse ordering to get min-heap behavior.
impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .abs_tick
            .cmp(&self.abs_tick)
            .then_with(|| kind_priority(&other.kind).cmp(&kind_priority(&self.kind)))
    }
}
impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.abs_tick == other.abs_tick && kind_priority(&self.kind) == kind_priority(&other.kind)
    }
}
impl Eq for ScheduledEvent {}

struct Scheduler {
    tick_period: Duration,
    start_instant: Instant,

    heap: BinaryHeap<ScheduledEvent>,
    current: LoopData,

    // MIDI clock config
    send_midi_clock: bool,
    ticks_per_midi_clock: u64,

    boundary_tx: Option<SyncSender<()>>,
}

impl Scheduler {
    fn new(
        loopdata: LoopData,
        _tpq: u64,
        _bpm: f64,
        tick_period: Duration,
        start_instant: Instant,
        send_midi_clock: bool,
        ticks_per_midi_clock: u64,
        boundary_tx: Option<SyncSender<()>>,
    ) -> Self {
        let mut s = Self {
            tick_period,
            start_instant,
            heap: BinaryHeap::new(),
            current: loopdata,
            send_midi_clock,
            ticks_per_midi_clock: ticks_per_midi_clock.max(1),
            boundary_tx: boundary_tx,
        };

        // Build schedule anchored at tick 0
        s.rebuild_heap_at_boundary(0);

        // Add MIDI clock stream (anchored at tick 0)
        if s.send_midi_clock {
            s.heap.push(ScheduledEvent {
                abs_tick: 0,
                kind: EventKind::MidiClock,
                period: s.ticks_per_midi_clock,
            });
        }

        s
    }

    fn now_tick(&self) -> u64 {
        let elapsed = self.start_instant.elapsed();
        let e_ns = elapsed.as_nanos() as u128;
        let t_ns = self.tick_period.as_nanos() as u128;
        if t_ns == 0 {
            return 0;
        }
        (e_ns / t_ns) as u64
    }

    fn tick_to_instant(&self, abs_tick: u64) -> Instant {
        // start_instant + abs_tick * tick_period, via nanos
        let tick_ns = self.tick_period.as_nanos() as f64;
        let target_ns = (abs_tick as f64 * tick_ns).round().max(0.0) as u128;
        self.start_instant + Duration::from_nanos(target_ns.min(u128::from(u64::MAX)) as u64)
    }

    fn wait_until_next_deadline(&self) {
        let Some(next_ev) = self.heap.peek() else {
            thread::sleep(Duration::from_millis(1));
            return;
        };

        let deadline = self.tick_to_instant(next_ev.abs_tick);

        // Sleep-until-near, then spin for the last few hundred microseconds.
        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline - now;

            if remaining > Duration::from_micros(800) {
                thread::sleep(remaining - Duration::from_micros(500));
            } else {
                std::hint::spin_loop();
            }
        }
    }

    fn process_due_events(
        &mut self,
        conn_out: &mut midir::MidiOutputConnection,
        pending: &mut Option<LoopData>,
    ) {
        let now_tick = self.now_tick();

        while let Some(ev) = self.heap.peek().copied() {
            if ev.abs_tick > now_tick {
                break;
            }
            let ev = self.heap.pop().unwrap();

            match ev.kind {
                EventKind::NoteOn { ch, note, vel } => note_on(conn_out, ch, note, vel),
                EventKind::NoteOff { ch, note } => note_off(conn_out, ch, note),
                EventKind::MidiClock => {
                    let _ = conn_out.send(&[MIDI_CLOCK]);
                }
                EventKind::Boundary => {
                    // trigger producer once per loop start (non-blocking)
                    if let Some(tx) = &self.boundary_tx {
                        let _ = tx.try_send(());
                    }
                    // swap at boundary if a new loop arrived
                    if let Some(new_loop) = pending.take() {
                        all_notes_off(conn_out);
                        self.current = new_loop;
                        self.rebuild_heap_at_boundary(ev.abs_tick);

                        // NOTE: MIDI clock stream stays in heap (independent), so we don't clear it here.
                        // rebuild_heap_at_boundary() only rebuilds note/boundary events; it does not touch MidiClock.
                        continue;
                    }
                }
            }

            if ev.period > 0 {
                let mut next = ev;
                next.abs_tick = next.abs_tick.saturating_add(ev.period);
                self.heap.push(next);
            }
        }
    }

    fn rebuild_heap_at_boundary(&mut self, boundary_abs_tick: u64) {
        // Keep MIDI clock events, clear everything else
        let mut clock_events: Vec<ScheduledEvent> = Vec::new();
        if self.send_midi_clock {
            while let Some(ev) = self.heap.pop() {
                if matches!(ev.kind, EventKind::MidiClock) {
                    clock_events.push(ev);
                }
            }
        } else {
            self.heap.clear();
        }

        // Put clock events back
        for ev in clock_events {
            self.heap.push(ev);
        }

        let loop_len = self.current.loop_len_ticks.max(1);

        // Boundary event repeats every loop
        self.heap.push(ScheduledEvent {
            abs_tick: boundary_abs_tick,
            kind: EventKind::Boundary,
            period: loop_len,
        });

        for tr in &self.current.tracks {
            let ch = tr.channel & 0x0F;
            let vel = tr.velocity;
            let default_note = tr.default_note;

            let mut events = tr.events.clone();
            events.sort_by_key(|e| e.start_tick);

            for e in events {
                let start = e.start_tick % loop_len;
                let note = if e.note == 0 { default_note } else { e.note };
                let len = e.length_ticks.max(1);

                let on_abs = boundary_abs_tick + start;
                let off_abs = boundary_abs_tick + start + len;

                self.heap.push(ScheduledEvent {
                    abs_tick: on_abs,
                    kind: EventKind::NoteOn { ch, note, vel },
                    period: loop_len,
                });

                self.heap.push(ScheduledEvent {
                    abs_tick: off_abs,
                    kind: EventKind::NoteOff { ch, note },
                    period: loop_len,
                });
            }
        }
    }
}

// ======================= timing helpers =======================

fn tick_period_from_bpm_tpq(bpm: f64, tpq: u64) -> Duration {
    let secs = 60.0 / (bpm * tpq as f64);
    Duration::from_secs_f64(secs)
}

// ======================= MIDI helpers =======================

fn send_realtime(conn_out: &mut midir::MidiOutputConnection, status: u8) {
    let _ = conn_out.send(&[status]);
}

fn note_on(conn_out: &mut midir::MidiOutputConnection, ch: u8, note: u8, vel: u8) {
    let status = 0x90 | (ch & 0x0F);
    let _ = conn_out.send(&[status, note, vel]);
}

fn note_off(conn_out: &mut midir::MidiOutputConnection, ch: u8, note: u8) {
    let status = 0x80 | (ch & 0x0F);
    let _ = conn_out.send(&[status, note, 0]);
}

fn all_notes_off(conn_out: &mut midir::MidiOutputConnection) {
    for ch in 0u8..16 {
        let status = 0xB0 | ch;
        let _ = conn_out.send(&[status, 123, 0]);
    }
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
