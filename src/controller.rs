use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::{collections::HashMap, fs, path::Path};

use crate::midievol::{MidievolConfig, ModFunc};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Domain {
    pub cc: u8,
    pub value: u8,
    pub funcs: HashMap<String, FuncConfig>,
    pub steps: BTreeMap<u32, BTreeMap<String, StepFunc>>,
    #[serde(skip)]
    pub filename: String,
}

impl Domain {
    pub fn get_values(&self) -> Option<BTreeMap<String, StepFunc>> {
        let n = self.steps.len();
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

        let mut ret: BTreeMap<String, StepFunc> = BTreeMap::new();

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

    // /// Gets step by 0-based index from YAML keys like "1", "2", ...
    // fn get_step_by_index(&self, idx0: usize) -> Option<&BTreeMap<String, StepFunc>> {
    //     // Your YAML uses "1" as first step; map 0 -> "1"
    //     let key: u32 = idx0.try_into().unwrap();
    //     self.steps.get(&key)
    // }

    fn get_step_by_index(&self, idx: usize) -> Option<&BTreeMap<String, StepFunc>> {
        self.steps.iter().nth(idx).map(|(_, v)| v)
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

pub fn load_domains(folder_path: impl AsRef<Path>) -> anyhow::Result<Vec<Domain>> {
    let folder_path = folder_path.as_ref();

    if !folder_path.is_dir() {
        anyhow::bail!("Provided path is not a directory: {:?}", folder_path);
    }

    // 1️⃣ Collect YAML file paths first
    let mut paths: Vec<PathBuf> = fs::read_dir(folder_path)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            if !path.is_file() {
                return None;
            }

            let is_yaml = matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("yaml" | "yml")
            );

            if is_yaml { Some(path) } else { None }
        })
        .collect();

    // 2️⃣ Sort alphabetically by filename
    paths.sort_by(|a, b| a.file_name().unwrap().cmp(b.file_name().unwrap()));

    // 3️⃣ Load in sorted order
    let mut domains = Vec::with_capacity(paths.len());

    for path in paths {
        let yaml_str = fs::read_to_string(&path)?;
        let mut domain: Domain = serde_yaml::from_str(&yaml_str)?;

        domain.filename = path.to_string_lossy().to_string();
        domains.push(domain);
    }

    Ok(domains)
}

#[derive(Clone, Debug)]
pub struct Controller {
    domains: Vec<Domain>,
}

impl Controller {
    pub fn new(domains: Vec<Domain>) -> Self {
        Self { domains }
    }

    pub fn reload_domains(&mut self, domains: Vec<Domain>) -> () {
        self.domains = domains;
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

    pub fn domains_count(self) -> usize {
        self.domains.len()
    }
}

impl Controller {
    pub fn save_step(&mut self, domain_idx: usize, step: usize, funcs: Vec<ModFunc>) -> Result<()> {
        let domain = self
            .domains
            .get_mut(domain_idx)
            .ok_or_else(|| anyhow!("domain index out of range: {}", domain_idx))?;

        if step == 0 {
            return Err(anyhow!("step must be 1-based (got 0)"));
        }

        // YAML keys are "1", "2", ...
        let step_key = step.try_into().unwrap();

        // Build the step map from CURRENT modfunc values, but only for funcs this domain knows about.
        let mut step_map: BTreeMap<String, StepFunc> = BTreeMap::new();

        for func_name in domain.funcs.keys() {
            let mf = funcs
                .iter()
                .find(|m| &m.name == func_name)
                .ok_or_else(|| anyhow!("modfunc '{}' not found in provided funcs", func_name))?;

            // Convert ModFunc -> StepFunc
            let step_func = StepFunc {
                weight: mf.weight,                                   // assuming f64
                params: mf.params.iter().map(|p| p.value).collect(), // assuming param.value: f64
                voices: mf.voices,                                   // [bool; 3]
                split_voices: mf.split_voices,
            };

            step_map.insert(func_name.clone(), step_func);
        }

        // Ensure steps map exists and set/overwrite this step
        domain.steps.insert(step_key, step_map);

        // Write YAML back to the same file
        if domain.filename.is_empty() {
            return Err(anyhow!(
                "domain.filename is empty; set it in load_domains() so saving knows where to write"
            ));
        }

        let yaml_out = serde_yaml::to_string(&domain)?;
        fs::write(&domain.filename, yaml_out)?;

        Ok(())
    }
}

fn apply_values_to_modfunc(mf: &mut ModFunc, v: &StepFunc) {
    mf.weight = v.weight;

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
