use crate::{
    error::{check_null, check_null_mut, Error},
    tflite_sys,
};

/// A list of edge TPU devices.
pub(crate) struct Devices {
    /// SAFETY: This pointer is owned by the caller that constructs Devices
    /// and is never mutated externally.
    devices: *const tflite_sys::edgetpu_device,
    len: usize,
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
        Ok(Self { devices, len })
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
        let devices = self.devices;
        (0..self.len()).map(move |offset| {
            // SAFETY: devices is guaranteed to be valid, and pointing to data with offset < len
            Ok(unsafe { *check_null(devices.add(offset)).ok_or(Error::GetDevicePtr)? }.type_)
        })
    }
}

impl Drop for Devices {
    fn drop(&mut self) {
        // SAFETY: self.devices is a valid input.
        unsafe {
            tflite_sys::edgetpu_free_devices(self.devices as _);
        }
    }
}
