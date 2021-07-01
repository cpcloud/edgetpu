use crate::{error::Error, ffi::ffi};
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
        let contexts = ffi::get_all_device_infos()
            .map_err(Error::GetAllDeviceInfos)?
            .into_iter()
            .map(|ffi::DeviceInfo { typ, path }| {
                ffi::make_edge_tpu_context(typ, &path).map_err(Error::MakeEdgeTpuContext)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let len = contexts.len();
        Ok(Self {
            contexts: Arc::new(contexts),
            allocated: Arc::new(Mutex::new(bitbox![0; len])),
        })
    }

    /// Allocate a single TPU device from the pool of devices.
    #[instrument(name = "Devices::allocate_one", skip(self))]
    pub(crate) fn allocate_one(&self) -> Result<SharedPtr<ffi::EdgeTpuContext>, Error> {
        self.contexts
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
            .ok_or(Error::FindEdgeTpuDevice)
    }
}
