use crate::midievol;

#[derive(Clone, Debug)]
pub struct ConfigReconciler {
    pub active: midievol::MidievolConfig,
    pub target: midievol::MidievolConfig,
}

impl ConfigReconciler {
    pub fn new(initial: midievol::MidievolConfig) -> Self {
        Self {
            active: initial.clone(),
            target: initial,
        }
    }

    // /// Update the target immediately (user intent changes instantly).
    // pub fn set_target(&mut self, new_target: midievol::MidievolConfig) {
    //     self.target = new_target;
    // }

    /// Call every loop tick (or at least frequently).
    /// Returns true if active config changed.
    pub fn step(&mut self) -> bool {
        if self.active.approx_eq(&self.target, 1e-6) {
            return false;
        }

        self.active = self.target.clone();

        true
    }
}
