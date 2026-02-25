use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path};

use crate::midievol::{MidievolConfig, ModFunc};

#[derive(Debug, Serialize, Deserialize)]
pub struct Domain {
    pub cc: u8,
    pub value: u8,
    pub funcs: HashMap<String, FuncConfig>,
    pub steps_count: u32,
    pub steps: HashMap<String, HashMap<String, StepFunc>>,
}

impl Domain {
    pub fn get_values(&self) -> Option<HashMap<String, StepFunc>> {
        let n = self.steps_count as usize;
        if n == 0 {
            return None;
        }
        if n == 1 {
            return self.get_step_by_index(0).cloned();
        }

        // Clamp value to [0, 127]
        let v: f64 = self.value.clamp(0, 127).into();

        // Map 0..127 to step space 0..(n-1)
        let pos: f64 = (v / 127.0) * ((n - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let t = (pos - (lo as f64)).clamp(0.0, 1.0);

        let s_lo = self.get_step_by_index(lo)?;
        let s_hi = self.get_step_by_index(hi)?;

        // Helper: choose closest step for "none"
        let pick_hi = t >= 0.5;

        let mut ret: HashMap<String, StepFunc> = HashMap::new();

        for func_name in self.funcs.keys() {
            let func = self.funcs.get(func_name)?;
            let s_lo = s_lo.get(func_name).unwrap();
            let s_hi = s_hi.get(func_name).unwrap();

            // Weight
            let weight = match func.weight {
                WeightMode::Linear => lerp(s_lo.weight, s_hi.weight, t),
                WeightMode::None => {
                    if pick_hi {
                        s_hi.weight
                    } else {
                        s_lo.weight
                    }
                }
            };

            // Params: per-index mode, default to None if not provided
            let max_params = s_lo.params.len().max(s_hi.params.len());
            let mut params = Vec::with_capacity(max_params);

            for i in 0..max_params {
                let a = s_lo.params.get(i).map(|p| *p).unwrap_or(0.0);
                let b = s_hi.params.get(i).map(|p| *p).unwrap_or(0.0);

                let mode = func.params.get(i).cloned().unwrap_or(ParamMode::None);

                let value = match mode {
                    ParamMode::Linear => lerp(a, b, t),
                    ParamMode::None => {
                        if pick_hi {
                            b
                        } else {
                            a
                        }
                    }
                };

                params.push(value);
            }

            // Non-numeric fields: pick closest step (you can change this rule)
            let (voices, split_voices) = if pick_hi {
                (s_hi.voices.clone(), s_hi.split_voices)
            } else {
                (s_lo.voices.clone(), s_lo.split_voices)
            };

            ret.insert(
                func_name.to_string(),
                StepFunc {
                    weight,
                    params,
                    voices,
                    split_voices,
                },
            );
        }
        Some(ret)
    }

    /// Gets step by 0-based index from YAML keys like "1", "2", ...
    fn get_step_by_index(&self, idx0: usize) -> Option<&HashMap<String, StepFunc>> {
        // Your YAML uses "1" as first step; map 0 -> "1"
        let key = (idx0 + 1).to_string();
        self.steps.get(&key)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FuncConfig {
    pub weight: WeightMode,
    pub params: Vec<ParamMode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightMode {
    None,
    Linear,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamMode {
    Linear,
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepFunc {
    pub weight: f64,
    pub params: Vec<f64>,
    pub voices: [bool; 3],
    #[serde(rename = "splitVoices")]
    pub split_voices: bool,
}

pub struct Controller {
    domains: Vec<Domain>,
}

impl Controller {
    pub fn new(domains: Vec<Domain>) -> Self {
        Self { domains }
    }

    pub fn load_domains(&mut self, folder_path: impl AsRef<Path>) -> anyhow::Result<()> {
        let folder_path = folder_path.as_ref();

        if !folder_path.is_dir() {
            anyhow::bail!("Provided path is not a directory: {:?}", folder_path);
        }

        for entry in fs::read_dir(folder_path)? {
            let entry = entry?;
            let path = entry.path();

            // Skip directories
            if !path.is_file() {
                continue;
            }

            // Only load .yaml or .yml files
            let is_yaml = match path.extension().and_then(|e| e.to_str()) {
                Some("yaml") | Some("yml") => true,
                _ => false,
            };

            if !is_yaml {
                continue;
            }

            let yaml_str = fs::read_to_string(&path)?;
            let domain: Domain = serde_yaml::from_str(&yaml_str)?;

            self.domains.push(domain);
        }

        Ok(())
    }

    pub fn apply_domains(&self, cfg: &mut MidievolConfig) {
        for domain in &self.domains {
            let Some(values_by_name) = domain.get_values() else {
                continue;
            };

            for (name, values) in values_by_name {
                if let Some(mf) = cfg.modfuncs.iter_mut().find(|m| m.name == name) {
                    apply_values_to_modfunc(mf, &values);
                }
            }
        }
    }

    pub fn update_domain(&mut self, cc: u8, value: u8) -> Option<usize> {
        if let Some(idx) = self.domains.iter().position(|d| d.cc == cc) {
            self.domains[idx].value = value.into();
            Some(idx)
        } else {
            None
        }
    }
}

fn apply_values_to_modfunc(mf: &mut ModFunc, v: &StepFunc) {
    mf.weight = v.weight.round();

    for (idx, param) in v.params.iter().enumerate() {
        match mf.params[idx].t {
            crate::midievol::ModFuncParamType::Note => mf.params[idx].value = (*param).round(),
            crate::midievol::ModFuncParamType::Float => mf.params[idx].value = *param,
            crate::midievol::ModFuncParamType::Int => mf.params[idx].value = (*param).round(),
            crate::midievol::ModFuncParamType::Bool => mf.params[idx].value = (*param).round(),
        };
    }

    mf.voices = v.voices;
    mf.split_voices = v.split_voices;
}
