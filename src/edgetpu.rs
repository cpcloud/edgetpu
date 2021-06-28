use crate::{coral_ffi::ffi, error::Error};
use bitvec::{bitbox, prelude::BitBox};
use cxx::SharedPtr;
use std::sync::{Arc, Mutex};
use tracing::instrument;

/// A list of edge TPU devices.
#[derive(Clone)]
pub(crate) struct Devices {
    contexts: Arc<Vec<SharedPtr<ffi::EdgeTpuContext>>>,
    allocated: Arc<Mutex<BitBox>>,
}

impl Devices {
    /// Construct a list of edgetpu devices.
    pub(crate) fn new() -> Result<Self, Error> {
        let contexts = Arc::new(
            ffi::get_all_device_infos()
                .into_iter()
                .map(|ffi::DeviceInfo { typ, path }| unsafe {
                    ffi::make_edge_tpu_context(typ, &path)
                })
                .collect::<Vec<_>>(),
        );
        let len = contexts.len();
        Ok(Self {
            contexts,
            allocated: Arc::new(Mutex::new(bitbox![0; len])),
        })
    }

    /// Return the number of devices.
    pub(crate) fn len(&self) -> usize {
        self.contexts.len()
    }

    /// Allocate a single TPU device from the pool of devices.
    #[instrument(name = "Devices::allocate_one", skip(self))]
    pub(crate) fn allocate_one(&self) -> Result<SharedPtr<ffi::EdgeTpuContext>, Error> {
        Ok(self
            .contexts
            .iter()
            .cloned()
            .enumerate()
            .find_map(|(index, context)| {
                let mut allocated = self.allocated.lock().unwrap();
                if !allocated[index] {
                    allocated.set(index, true);
                    Some(context)
                } else {
                    None
                }
            })
            .ok_or(Error::FindEdgeTpuDevice)?)
    }
}
