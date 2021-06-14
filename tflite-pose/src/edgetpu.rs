use crate::{error::Error, tflite_sys};
use std::marker::PhantomData;

pub(crate) struct Devices<'a> {
    devices: *mut tflite_sys::edgetpu_device,
    len: usize,
    _p: PhantomData<&'a ()>,
}

pub(crate) struct Device<'a> {
    device: *mut tflite_sys::edgetpu_device,
    _p: PhantomData<&'a ()>,
}

impl<'a> Device<'a> {
    fn new(device: *mut tflite_sys::edgetpu_device) -> Result<Self, Error> {
        if device.is_null() {
            return Err(Error::GetDevice);
        }
        Ok(Self {
            device,
            _p: PhantomData,
        })
    }

    pub(crate) fn delegate<'b>(&'a self) -> Result<Delegate<'b>, Error> {
        let delegate = unsafe {
            tflite_sys::edgetpu_create_delegate(
                (*self.device).type_,
                std::ptr::null(),
                std::ptr::null(),
                0,
            )
        };

        if delegate.is_null() {
            return Err(Error::CreateEdgeTpuDelegate);
        }

        Ok(Delegate {
            delegate,
            _p: PhantomData,
        })
    }
}

impl<'a> Devices<'a> {
    pub(crate) fn new() -> Result<Self, Error> {
        let mut len: usize = 0;

        // # SAFETY: check for nullness after return; len is guaranteed to point to valid data
        let devices = unsafe { tflite_sys::edgetpu_list_devices(&mut len) };
        if devices.is_null() {
            return Err(Error::ListDevices);
        }

        Ok(Self {
            devices,
            len,
            _p: PhantomData,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn iter(&'a self) -> impl Iterator<Item = Result<Device<'a>, Error>> {
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

pub(crate) struct Delegate<'del> {
    pub(crate) delegate: *mut tflite_sys::TfLiteDelegate,
    _p: PhantomData<&'del ()>,
}

impl<'del> Delegate<'del> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut tflite_sys::TfLiteDelegate {
        self.delegate
    }
}

impl<'a> Drop for Delegate<'a> {
    fn drop(&mut self) {
        // # SAFETY: self.delegate is guaranteed to be valid
        unsafe {
            tflite_sys::edgetpu_free_delegate(self.delegate);
        }
    }
}
