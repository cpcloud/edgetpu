use crate::{error::Error, tflite_sys};

pub(crate) struct Devices<'a> {
    devices: *mut tflite_sys::edgetpu_device,
    len: usize,
    _p: std::marker::PhantomData<&'a ()>,
}

pub(crate) struct Device<'a> {
    device: *mut tflite_sys::edgetpu_device,
    _p: std::marker::PhantomData<&'a ()>,
}

impl<'a> Device<'a> {
    fn new(device: *mut tflite_sys::edgetpu_device) -> Result<Self, Error> {
        if device.is_null() {
            return Err(Error::GetDevice);
        }
        Ok(Self {
            device,
            _p: std::marker::PhantomData,
        })
    }

    pub(crate) fn r#type(&self) -> tflite_sys::edgetpu_device_type {
        // SAFETY: self.device is guaranteed to be non-null
        unsafe { *self.device }.type_
    }

    pub(crate) fn path(&self) -> Result<&str, std::str::Utf8Error> {
        // SAFETY: self.device is guaranteed to be non-null
        unsafe { std::ffi::CStr::from_ptr((*self.device).path) }.to_str()
    }
}

impl<'a> Devices<'a> {
    pub(crate) fn new() -> Result<Self, Error> {
        let mut len: usize = 0;
        let devices = unsafe { tflite_sys::edgetpu_list_devices(&mut len) };
        if devices.is_null() {
            return Err(Error::ListDevices);
        }
        Ok(Self {
            devices,
            len,
            _p: std::marker::PhantomData,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn iter(&'a mut self) -> impl Iterator<Item = Result<Device<'a>, Error>> {
        // SAFETY: self.devices + 0..self.len() is guaranteed to be non-null
        let devices = self.devices;
        (0..self.len()).map(move |offset| Device::new(unsafe { devices.add(offset) }))
    }
}

impl<'a> Drop for Devices<'a> {
    fn drop(&mut self) {
        unsafe {
            tflite_sys::edgetpu_free_devices(self.devices);
        }
    }
}
