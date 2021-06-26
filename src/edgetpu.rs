use std::sync::{
    atomic::{AtomicPtr, Ordering},
    Arc, RwLock,
};

use crate::{
    error::{check_null, check_null_mut, Error},
    tflite_sys,
};
use bitvec::{bitbox, prelude::BitBox};

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
    allocd: Arc<RwLock<BitBox>>,
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
        Ok(Self {
            devices: Arc::new(RawDevices(AtomicPtr::new(devices))),
            len,
            allocd: Arc::new(RwLock::new(bitbox![0; len])),
        })
    }

    /// Return the number of devices.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Return whether there are no devices.
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Construct an iterator over device types.
    pub(crate) fn types(
        &self,
    ) -> impl Iterator<Item = Result<tflite_sys::edgetpu_device_type, Error>> {
        let devices = self.devices.clone();
        (0..self.len()).map(move |offset| {
            // SAFETY: devices is guaranteed to be valid, and pointing to data with offset < len
            Ok(unsafe {
                *check_null(devices.0.load(Ordering::SeqCst).add(offset))
                    .ok_or(Error::GetDevicePtr)?
            }
            .type_)
        })
    }

    /// Return an iterator over all devices not currently allocated to an interpreter.
    fn unallocated(
        &self,
    ) -> impl Iterator<Item = Result<(usize, tflite_sys::edgetpu_device_type), Error>> + '_ {
        let allocd = self.allocd.clone();
        self.types().enumerate().filter_map(move |(index, r#type)| {
            if !allocd.read().unwrap()[index] {
                Some(r#type.map(|t| (index, t)))
            } else {
                None
            }
        })
    }

    /// Allocate a single TPU device from the pool of devices.
    pub(crate) fn allocate_one(&mut self) -> Result<tflite_sys::edgetpu_device_type, Error> {
        let (index, r#type) = self.unallocated().next().ok_or(Error::GetEdgeTpuDevice)??;
        if self.allocd.read().unwrap()[index] {
            return Err(Error::AllocateEdgeTpu(index));
        }
        self.allocd.write().unwrap().set(index, true);
        Ok(r#type)
    }
}
