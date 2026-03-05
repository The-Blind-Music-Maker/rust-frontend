use anyhow::{Result, anyhow};
use std::{collections::HashMap, fs};

use crate::midievol::{MidievolConfig, Score};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScoreConfig {
    /// Per modfunc: 3 thresholds => 4 buckets
    /// (t0, t1, t2) means:
    /// score < t0 => 0 (really bad)
    /// score < t1 => 1 (bad)
    /// score < t2 => 2 (good)
    /// else       => 3 (really good)
    pub funcs: HashMap<String, [f64; 3]>,
}

impl ScoreConfig {
    /// Categorize a single raw score into 4 buckets [0..=3] using thresholds.
    #[inline]
    pub fn category_for(&self, func_name: &str, score: f64) -> Result<u8> {
        let [t0, t1, t2] = *self
            .funcs
            .get(func_name)
            .ok_or_else(|| anyhow!("missing thresholds for func '{func_name}'"))?;

        if !(t0 < t1 && t1 < t2) {
            return Err(anyhow!(
                "invalid thresholds for '{func_name}': must satisfy t0 < t1 < t2, got [{t0}, {t1}, {t2}]"
            ));
        }

        let cat = if score < t0 {
            0
        } else if score < t1 {
            1
        } else if score < t2 {
            2
        } else {
            3
        };
        Ok(cat)
    }

    /// Combine per-func categories using the modfunc weights and return a single bucket [0..=3].
    ///
    /// - Uses cfg.modfuncs for weights and names
    /// - Uses `scores_by_name` for the actual score values per modfunc name
    /// - Skips modfuncs that have no score entry (or you can make that an error; see comment)
    pub fn combined_ranking(
        &self,
        cfg: &MidievolConfig,
        scores_by_name: HashMap<String, f64>,
    ) -> Result<u8> {
        let mut weighted_sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for mf in &cfg.modfuncs {
            if mf.weight.is_sign_negative() {
                return Err(anyhow!(
                    "negative weight for modfunc '{}': {}",
                    mf.name,
                    mf.weight
                ));
            }

            // If you want missing scores to be a hard error, replace this `if let` with `let score = ...?;`
            let Some(&score) = scores_by_name.get(&mf.name) else {
                continue;
            };

            let cat = self.category_for(&mf.name, score)? as f64;

            // Treat 0-weight funcs as non-contributing.
            if mf.weight > 0.0 {
                weighted_sum += cat * mf.weight;
                weight_sum += mf.weight;
            }
        }

        if weight_sum <= 0.0 {
            return Err(anyhow!(
                "cannot compute combined ranking: total contributing weight is 0 (no scores and/or all weights 0)"
            ));
        }

        // Weighted average in [0.0 .. 3.0]
        let avg = weighted_sum / weight_sum;

        // Map average to 4 buckets.
        // Midpoints: 0.5, 1.5, 2.5
        let combined = if avg < 0.5 {
            0
        } else if avg < 1.5 {
            1
        } else if avg < 2.5 {
            2
        } else {
            3
        };

        Ok(combined)
    }
}

/// Optional helper for display
pub fn ranking_label(r: u8) -> &'static str {
    match r {
        0 => "really bad",
        1 => "bad",
        2 => "good",
        3 => "really good",
        _ => "unknown",
    }
}

pub fn load_score_config(path: &str) -> anyhow::Result<ScoreConfig> {
    let yaml_str = fs::read_to_string(path)?;
    let cfg: ScoreConfig = serde_yaml::from_str(&yaml_str)?;
    Ok(cfg)
}

pub fn scores_to_map(cfg: &MidievolConfig, scores: &[Option<Score>]) -> HashMap<String, f64> {
    cfg.modfuncs
        .iter()
        .zip(scores.iter())
        .filter_map(|(mf, score_opt)| score_opt.as_ref().map(|s| (mf.name.clone(), s.score)))
        .collect()
}
