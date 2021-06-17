use crate::{
    error::{check_null_mut, Error},
    tflite::Delegate,
    tflite_sys,
};

/// A list of coral edge TPU devices
pub(crate) struct Devices {
    /// SAFETY: This pointer is owned by the constructor and never mutated
    devices: *const tflite_sys::edgetpu_device,
    len: usize,
}

pub(crate) struct Device<'parent> {
    device: &'parent tflite_sys::edgetpu_device,
}

impl<'parent> Device<'parent> {
    fn new(device: &'parent tflite_sys::edgetpu_device) -> Self {
        Self { device }
    }

    pub(crate) fn delegate(&self) -> Result<Delegate, Error> {
        Ok(Delegate::new(
            check_null_mut(
                // SAFETY: inputs are all valid, and the return value is checked for null
                unsafe {
                    tflite_sys::edgetpu_create_delegate(
                        (*self.device).type_,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                    )
                },
                || Error::CreateEdgeTpuDelegate,
            )?,
            |delegate| unsafe { tflite_sys::edgetpu_free_delegate(delegate) },
        ))
    }
}

impl Devices {
    pub(crate) fn new() -> Result<Self, Error> {
        let mut len: usize = 0;
        let devices = check_null_mut(
            // SAFETY: len is guaranteed to point to valid data (and is checked by the implementation)
            unsafe { tflite_sys::edgetpu_list_devices(&mut len) },
            || Error::ListDevices,
        )?;

        Ok(Self { devices, len })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = Device> {
        // SAFETY: self.devices + 0..self.len() is guaranteed to be non-null
        let devices = self.devices;
        (0..self.len()).map(move |offset| {
            Device::new(unsafe { devices.add(offset).as_ref().expect("device is null") })
        })
    }
}

impl Drop for Devices {
    fn drop(&mut self) {
        // SAFETY: self.devices is valid
        unsafe {
            tflite_sys::edgetpu_free_devices(self.devices as _);
        }
    }
}
