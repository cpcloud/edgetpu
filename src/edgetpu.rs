use crate::{
    error::{check_null, check_null_mut, Error},
    tflite_sys,
};
use bitvec::{bitbox, prelude::BitBox};
use std::{
    ffi::CStr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, Mutex,
    },
};
use tracing::{info, instrument};

struct RawDevices(AtomicPtr<tflite_sys::edgetpu_device>);

impl Drop for RawDevices {
    fn drop(&mut self) {
        // SAFETY: self.devices is a valid input.
        unsafe {
            tflite_sys::edgetpu_free_devices(self.0.load(Ordering::SeqCst));
        }
    }
}

/// A list of edge TPU devices.
#[derive(Clone)]
pub(crate) struct Devices {
    /// SAFETY: This pointer is owned by the caller that constructs Devices
    /// and is never mutated externally.
    devices: Arc<RawDevices>,
    len: usize,
    allocated: Arc<Mutex<BitBox>>,
}

impl Devices {
    /// Construct a list of edgetpu devices.
    pub(crate) fn new() -> Result<Self, Error> {
        let mut len: usize = 0;
        let devices = check_null_mut(
            // SAFETY: len is guaranteed to point to valid data (and is checked by the implementation)
            unsafe { tflite_sys::edgetpu_list_devices(&mut len) },
        )
        .ok_or(Error::ListDevices)?;
        if len == 0 {
            return Err(Error::GetEdgeTpuDevice);
        }
        Ok(Self {
            devices: Arc::new(RawDevices(AtomicPtr::new(devices))),
            len,
            allocated: Arc::new(Mutex::new(bitbox![0; len])),
        })
    }

    /// Return the number of devices.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Construct an iterator over device types.
    fn devices(
        &self,
    ) -> impl Iterator<Item = Result<(tflite_sys::edgetpu_device_type, &CStr), Error>> {
        (0..self.len()).map(move |offset| {
            // SAFETY: devices is guaranteed to be valid, and pointing to data with offset < len
            let device = unsafe {
                *check_null(self.devices.0.load(Ordering::SeqCst).add(offset))
                    .ok_or(Error::GetDevicePtr)?
            };
            Ok((device.type_, unsafe { CStr::from_ptr(device.path) }))
        })
    }

    /// Allocate a single TPU device from the pool of devices.
    #[instrument(name = "Devices::allocate_one", skip(self))]
    pub(crate) fn allocate_one(&self) -> Result<(tflite_sys::edgetpu_device_type, &CStr), Error> {
        let (r#type, path) = self
            .devices()
            .enumerate()
            .find_map(|(index, device_info)| {
                let mut allocated = self.allocated.lock().unwrap();
                if !allocated[index] {
                    allocated.set(index, true);
                    Some(device_info)
                } else {
                    None
                }
            })
            .ok_or(Error::FindEdgeTpuDevice)??;
        info!(message = "allocated device", path = ?path.to_str().map_err(Error::GetEdgeTpuDevicePath)?);
        Ok((r#type, path))
    }
}
