use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;

/// Convert a non-negative integer to "base-4" using mapping:
/// 0 -> 'A', 1 -> 'G', 2 -> 'C', 3 -> 'T'
pub fn num_to_base4(mut num: u32) -> String {
    if num == 0 {
        return "0".to_string();
    }

    const BASE4_CHARS: [char; 4] = ['A', 'G', 'C', 'T'];

    let mut out: Vec<char> = Vec::new();
    while num > 0 {
        let digit = (num % 4) as usize;
        out.push(BASE4_CHARS[digit]);
        num /= 4;
    }

    out.reverse();
    out.into_iter().collect()
}

/// Left-pad `s` with `pad_char` to total length `width`.
fn pad_start(mut s: String, width: usize, pad_char: char) -> String {
    if s.len() >= width {
        return s;
    }
    let mut padded = String::with_capacity(width);
    for _ in 0..(width - s.len()) {
        padded.push(pad_char);
    }
    padded.push_str(&s);
    s = padded;
    s
}

/// Equivalent to JS: Math.round(Math.random() * max)
fn rand_round_inclusive(rng: &mut impl Rng, max: u32) -> u32 {
    // JS Math.random() is [0, 1), then *max => [0, max), then round => integer 0..max (inclusive)
    let x: f64 = rng.random(); // [0.0, 1.0)
    (x * max as f64).round() as u32
}

fn get_random_note(rng: &mut impl Rng) -> u32 {
    rand_round_inclusive(rng, 840)
}

/// Creates one "note" DNA string:
/// - note:   random 0..=840   -> base4 -> pad to 8 with 'A'
/// - time?:  random 0..=2000  -> base4 -> pad to 8 with 'A'
/// - vel:    random 0..=127   -> base4 -> pad to 4 with 'A'
/// - dur?:   random 0..=4000  -> base4 -> pad to 16 with 'A'  (since 500*8=4000)
pub fn create_random_note(rng: &mut impl Rng) -> String {
    let a = pad_start(num_to_base4(get_random_note(rng)), 8, 'A');
    let b = pad_start(num_to_base4(rand_round_inclusive(rng, 2000)), 8, 'A');
    let c = pad_start(num_to_base4(rand_round_inclusive(rng, 127)), 4, 'A');
    let d = pad_start(num_to_base4(rand_round_inclusive(rng, 500 * 8)), 16, 'A');

    format!("{a}{b}{c}{d}")
}

/// Concatenate `num_notes` random notes into one DNA string.
pub fn create_random_melody(num_notes: usize, rng: &mut impl Rng) -> String {
    let mut dna = String::new();
    for _ in 0..num_notes {
        dna.push_str(&create_random_note(rng));
    }
    dna
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModFuncParamType {
    Note,
    Float,
    Int,
    Bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModFuncParam {
    pub name: String,
    pub range: [u8; 2],
    pub value: f64,

    #[serde(rename = "type")]
    pub t: ModFuncParamType,

    // NEW: CC number for this param
    #[serde(default)]
    pub cc: Option<u8>,

    // OPTIONAL: bind to a specific MIDI channel (0..15). If None => any channel.
    #[serde(default)]
    pub channel: Option<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModFunc {
    pub name: String,
    pub weight: f64,
    pub params: Vec<ModFuncParam>,

    #[serde(rename = "splitVoices")]
    pub split_voices: bool,

    #[serde(rename = "hasNormalizedScore")]
    pub has_normalized_score: bool,

    #[serde(rename = "normalizationFunc")]
    pub normalization_func: String,

    pub voices: [bool; 3],

    #[serde(rename = "scoreRange")]
    pub score_range: (Option<f64>, Option<f64>),

    // NEW: CC number for this modfunc weight
    #[serde(rename = "weightCC", default)]
    pub weight_cc: Option<u8>,

    // OPTIONAL: bind weight CC to a specific MIDI channel (0..15). If None => any channel.
    #[serde(rename = "weightChannel", default)]
    pub weight_channel: Option<u8>,
}

pub fn load_config(path: &str) -> Result<MidievolConfig, Box<dyn std::error::Error>> {
    let yaml_str = fs::read_to_string(path)?;
    let cfg: MidievolConfig = serde_yaml::from_str(&yaml_str)?;
    Ok(cfg)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Voices {
    pub min: u8,
    pub max: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MidievolConfig {
    pub voices: Voices,
    pub x_gens: u32,
    pub children: u32,
    pub bpm: f64,

    #[serde(rename = "modFuncs")]
    pub modfuncs: Vec<ModFunc>,
}

impl MidievolConfig {
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        voices_eq(&self.voices, &other.voices)
            && self.x_gens == other.x_gens
            && self.children == other.children
            && feq(self.bpm, other.bpm, tol)
            && vec_modfunc_eq(&self.modfuncs, &other.modfuncs, tol)
    }
}

fn voices_eq(a: &Voices, b: &Voices) -> bool {
    a.min == b.min && a.max == b.max
}

fn vec_modfunc_eq(a: &[ModFunc], b: &[ModFunc], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(ma, mb)| modfunc_eq(ma, mb, tol))
}

fn modfunc_eq(a: &ModFunc, b: &ModFunc, tol: f64) -> bool {
    a.name == b.name
        && feq(a.weight, b.weight, tol)
        && a.split_voices == b.split_voices
        && a.has_normalized_score == b.has_normalized_score
        && a.normalization_func == b.normalization_func
        && a.voices == b.voices
        && score_range_eq(a.score_range, b.score_range, tol)
        && a.weight_cc == b.weight_cc
        && a.weight_channel == b.weight_channel
        && vec_param_eq(&a.params, &b.params, tol)
}

fn vec_param_eq(a: &[ModFuncParam], b: &[ModFuncParam], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(pa, pb)| param_eq(pa, pb, tol))
}

fn param_eq(a: &ModFuncParam, b: &ModFuncParam, tol: f64) -> bool {
    a.name == b.name
        && a.range == b.range
        && feq(a.value, b.value, tol)
        && param_type_eq(&a.t, &b.t)
        && a.cc == b.cc
        && a.channel == b.channel
}

fn param_type_eq(a: &ModFuncParamType, b: &ModFuncParamType) -> bool {
    // enum; exact match is correct
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn score_range_eq(a: (Option<f64>, Option<f64>), b: (Option<f64>, Option<f64>), tol: f64) -> bool {
    opt_f64_eq(a.0, b.0, tol) && opt_f64_eq(a.1, b.1, tol)
}

fn opt_f64_eq(a: Option<f64>, b: Option<f64>, tol: f64) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => feq(x, y, tol),
        _ => false,
    }
}

/// A pretty robust float comparator:
/// - absolute tolerance near 0
/// - relative tolerance for larger magnitudes
/// - treats NaN as never equal (including NaN vs NaN)
fn feq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a == b {
        return true; // handles infinities too
    }
    let diff = (a - b).abs();
    if diff <= tol {
        return true;
    }
    // relative tolerance
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= tol * scale
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitPayload {
    pub dna: String,
    pub voices: Voices,

    #[serde(rename = "modFuncs")]
    pub modfuncs: Vec<ModFunc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvolvePayload {
    pub dna: String,
    pub x_gens: u32,
    pub children: u32,
    pub voices: Voices,

    #[serde(rename = "modFuncs")]
    pub modfuncs: Vec<ModFunc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Note {
    pub position: u32,
    pub pitch: u32,
    pub length: u32,
    pub volume: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreInfo {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Score {
    pub score: f64,
    pub info: Vec<ScoreInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Melody {
    pub notes: Vec<Note>,
    pub dna: String,
    pub scores_per_func: Vec<Option<Score>>,
    pub score: f64,
    pub bpm: u8,
}

pub fn send_init_req(
    endpoint: &str,
    payload: InitPayload,
) -> Result<Melody, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();

    let response = client.post(endpoint).json(&payload).send()?; // propagate errors

    // Optional: fail early on non-2xx responses
    if !response.status().is_success() {
        return Err(format!("Request failed: {}", response.status()).into());
    }

    let txt = response.text().unwrap();

    // Deserialize JSON body into Melody

    let melody: Melody = serde_json::from_str(&txt)?;

    Ok(melody)
}

pub fn send_evolve_req(
    endpoint: &str,
    payload: EvolvePayload,
) -> Result<Melody, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();

    let response = client.post(endpoint).json(&payload).send()?; // propagate errors

    // Optional: fail early on non-2xx responses
    if !response.status().is_success() {
        return Err(format!("Request failed: {}", response.status()).into());
    }

    let txt = response.text().unwrap();

    // Deserialize JSON body into Melody
    let melody: Melody = serde_json::from_str(&txt).unwrap(); //response.json()?;

    Ok(melody)
}

// use std::error::Error;

// #[tokio::main]
// async fn main() -> Result<(), Box<dyn Error>> {
//     let funcs = load_mod_funcs("mod_funcs.yaml")?;

//     // Debug: inspect JSON before sending
//     let json = serde_json::to_string_pretty(&funcs)?;
//     println!("{json}");

//     send_init_req("https://example.com/api/mod-funcs", funcs).await?;

//     Ok(())
// }
