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

    #[serde(rename = "modFuncs")]
    pub modfuncs: Vec<ModFunc>,
}

#[derive(Debug, Serialize, Deserialize)]
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
pub struct Melody {
    pub notes: Vec<Note>,
    pub dna: String,
    pub scores_per_func: Vec<Option<f64>>,
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
    let melody: Melody = serde_json::from_str(&txt).unwrap(); //response.json()?;

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
