use crate::{
    error::{check_null, check_null_mut, Error},
    tflite::Delegate,
    tflite_sys,
};

/// A list of coral edge TPU devices.
pub(crate) struct Devices {
    /// SAFETY: This pointer is owned by the caller that constructs Devices
    /// and is never mutated
    devices: *const tflite_sys::edgetpu_device,
    len: usize,
}

pub(crate) struct Device<'devices>(&'devices tflite_sys::edgetpu_device);

impl<'devices> Device<'devices> {
    fn new(device: &'devices tflite_sys::edgetpu_device) -> Self {
        Self(device)
    }

    pub(crate) fn delegate(&self) -> Result<Delegate, Error> {
        Delegate::new(
            check_null_mut(
                // SAFETY: inputs are all valid, and the return value is checked for null
                unsafe {
                    tflite_sys::edgetpu_create_delegate(
                        (*self.0).type_,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                    )
                },
            )
            .ok_or(Error::CreateEdgeTpuDelegate)?,
            |delegate| unsafe { tflite_sys::edgetpu_free_delegate(delegate) },
        )
    }
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

    /// Construct an iterator over devices.
    pub(crate) fn types(
        &self,
    ) -> impl Iterator<Item = Result<tflite_sys::edgetpu_device_type, Error>> {
        // SAFETY: self.devices + 0..self.len() is guaranteed to be non-null
        let devices = self.devices;
        (0..self.len()).map(move |offset| {
            // SAFETY: devices is guaranteed to be valid, and pointing to data with offset < len
            Ok(unsafe { *check_null(devices.add(offset), || Error::GetDevicePtr)? }.type_)
        })
    }
}

impl Drop for Devices {
    fn drop(&mut self) {
        // SAFETY: self.devices is a valid pointer.
        unsafe {
            tflite_sys::edgetpu_free_devices(self.devices as _);
        }
    }
}
