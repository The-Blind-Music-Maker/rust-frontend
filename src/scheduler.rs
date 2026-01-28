use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::mpsc::SyncSender;
use std::thread;
use std::time::{Duration, Instant};

const MIDI_CLOCK: u8 = 0xF8;

fn tick_period_from_bpm(tpq: u64, bpm: f64) -> Duration {
    let tpq = tpq.max(1) as f64;
    let bpm = bpm.max(1e-6); // avoid div-by-zero / nonsense

    // seconds per tick = 60 / (bpm * tpq)
    let secs_per_tick = 60.0 / (bpm * tpq);

    // convert to nanos (clamp to at least 1ns)
    let nanos = (secs_per_tick * 1_000_000_000.0).round().max(1.0) as u64;
    Duration::from_nanos(nanos)
}

// ======================= MIDI helpers =======================

pub fn send_realtime(conn_out: &mut midir::MidiOutputConnection, status: u8) {
    let _ = conn_out.send(&[status]);
}

pub fn note_on(conn_out: &mut midir::MidiOutputConnection, ch: u8, note: u8, vel: u8) {
    let status = 0x90 | (ch & 0x0F);
    let _ = conn_out.send(&[status, note, vel]);
}

pub fn send_cc(conn_out: &mut midir::MidiOutputConnection, ch: u8, cc: u8, value: u8) {
    let status = 0xB0 | (ch & 0x0F);
    let cc = cc & 0x7F; // ensure 0..127
    let value = value & 0x7F; // ensure 0..127
    let _ = conn_out.send(&[status, cc, value]);
}

pub fn note_off(conn_out: &mut midir::MidiOutputConnection, ch: u8, note: u8) {
    let status = 0x80 | (ch & 0x0F);
    let _ = conn_out.send(&[status, note, 0]);
}

pub fn all_notes_off(conn_out: &mut midir::MidiOutputConnection) {
    for ch in 0u8..16 {
        let status = 0xB0 | ch;
        let _ = conn_out.send(&[status, 123, 0]);
    }
}

#[derive(Clone, Debug)]
pub struct LoopData {
    pub loop_len_ticks: u64,
    pub tracks: Vec<TrackData>,
    pub voice_mediants: [u8; 3],
}

#[derive(Clone, Debug)]
pub struct TrackData {
    pub channel: u8,
    pub velocity: u8,
    pub default_note: u8,
    pub events: Vec<NoteEvent>,
}

#[derive(Clone, Copy, Debug)]
pub struct NoteEvent {
    pub start_tick: u64,   // position within loop [0..loop_len_ticks)
    pub note: u8,          // if 0, use track default_note
    pub length_ticks: u64, // duration in internal ticks
}

impl NoteEvent {
    pub fn note(start_tick: u64, note: u8, length_ticks: u64) -> Self {
        Self {
            start_tick,
            note,
            length_ticks,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum EventKind {
    NoteOn { ch: u8, note: u8, vel: u8 },
    NoteOff { ch: u8, note: u8 },
    Boundary,  // used to apply pending pattern swaps cleanly
    MidiClock, // MIDI realtime clock (0xF8)
}

#[derive(Clone, Copy, Debug)]
pub struct ScheduledEvent {
    abs_tick: u64,
    kind: EventKind,
    period: u64, // 0 => one-shot; otherwise reschedule by adding period
}

// Lower number = higher priority (processed first) when abs_tick ties
pub fn kind_priority(k: &EventKind) -> u8 {
    match k {
        EventKind::MidiClock => 0, // clock can be after note events if they coincide
        EventKind::Boundary => 1,  // apply swap before anything else at the boundary
        EventKind::NoteOff { .. } => 2, // turn off before on
        EventKind::NoteOn { .. } => 3,
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

pub struct Scheduler {
    tick_period: Duration,
    start_instant: Instant,

    heap: BinaryHeap<ScheduledEvent>,
    current: LoopData,

    // MIDI clock config
    send_midi_clock: bool,
    ticks_per_midi_clock: u64,

    boundary_tx: Option<SyncSender<()>>,

    tpq: u64,
    bpm: f64,
}

impl Scheduler {
    pub fn new(
        loopdata: LoopData,
        tpq: u64,
        bpm: f64,
        start_instant: Instant,
        send_midi_clock: bool,
        ticks_per_midi_clock: u64,
        boundary_tx: Option<SyncSender<()>>,
    ) -> Self {
        let mut s = Self {
            tick_period: tick_period_from_bpm(tpq, bpm),
            start_instant,
            heap: BinaryHeap::new(),
            current: loopdata,
            send_midi_clock,
            ticks_per_midi_clock: ticks_per_midi_clock.max(1),
            boundary_tx,

            tpq: tpq.max(1),
            bpm: bpm,
        };

        // ... unchanged
        s.rebuild_heap_at_boundary(0);

        if s.send_midi_clock {
            s.heap.push(ScheduledEvent {
                abs_tick: 0,
                kind: EventKind::MidiClock,
                period: s.ticks_per_midi_clock,
            });
        }

        s
    }

    pub fn set_bpm(&mut self, new_bpm: f64) {
        let new_period = tick_period_from_bpm(self.tpq, new_bpm);

        // compute where we are in "tick time" using the OLD period
        let now_tick = self.now_tick();

        // switch period
        self.tick_period = new_period;
        self.bpm = new_bpm;

        // re-anchor start_instant so that now_tick stays the same under the NEW period
        let tick_ns = self.tick_period.as_nanos() as u128;
        let back_ns = (now_tick as u128).saturating_mul(tick_ns);

        self.start_instant =
            Instant::now() - Duration::from_nanos(back_ns.min(u128::from(u64::MAX)) as u64);
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

    pub fn wait_until_next_deadline(&self) {
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

    pub fn process_due_events(
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
                        let mediants = &new_loop.voice_mediants.clone();
                        self.current = new_loop;
                        self.rebuild_heap_at_boundary(ev.abs_tick);

                        // We send the mediants as CC messages on channel 4
                        send_cc(conn_out, 1, 10, mediants[0]);
                        send_cc(conn_out, 2, 10, mediants[1]);
                        send_cc(conn_out, 3, 10, mediants[2]);

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
