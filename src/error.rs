use crate::tflite_sys;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("failed to list devices: got null pointer instead")]
    ListDevices,

    #[error("got null pointer for device")]
    GetDevicePtr,

    #[error("failed to convert Path to CString")]
    PathToCString(#[source] std::ffi::NulError),

    #[error("failed to convert usize to i32")]
    GetFfiIndex(#[source] std::num::TryFromIntError),

    #[error("failed to get input tensor: got null pointer instead")]
    GetInputTensor,

    #[error("failed to convert input tensor count i32 to usize")]
    ConvertInputTensorCount(#[source] std::num::TryFromIntError),

    #[error("failed to get input tensor at index {0}: only {1} input tensors available")]
    InputTensorIndex(usize, usize),

    #[error("failed to get output tensor: got null pointer instead")]
    GetOutputTensor,

    #[error("failed to convert output tensor count i32 to usize")]
    ConvertOutputTensorCount(#[source] std::num::TryFromIntError),

    #[error("expected {0} output tensors, got {1}")]
    GetExpectedNumOutputs(usize, usize),

    #[error("failed to get output tensor at index {0}: only {1} output tensors available")]
    OutputTensorIndex(usize, usize),

    #[error("failed to convert row count i32 to usize")]
    ConvertRowsToUsize,

    #[error("failed to get value of Mat::elemSize1")]
    GetElemSize1(#[source] opencv::Error),

    #[error("failed to construct array view from TfLiteTensor")]
    ConstructArrayView(#[source] ndarray::ShapeError),

    #[error("failed to convert usize value to keypoint kind: {0}")]
    ConvertUSizeToKeypointKind(usize),

    #[error("tflite error: {0}")]
    TfLite(&'static str),

    #[error("tflite delegate error: {0}")]
    Delegate(&'static str),

    #[error("tflite application error: {0}")]
    Application(&'static str),

    #[error("failed to construct model from file: C API returned null pointer")]
    GetModelFromFile,

    #[error("failed to create TfLiteInterpreterOptions structure")]
    CreateOptions,

    #[error("failed to create interpreter, got null pointer")]
    CreateInterpreter,

    #[cfg(feature = "gui")]
    #[error("failed to convert keypoint variant to usize: {0:?}")]
    KeypointVariantToUSize(crate::pose::KeypointKind),

    #[error("failed to create edgetpu delegate: got null pointer")]
    CreateEdgeTpuDelegate,

    #[cfg(feature = "posenet_decoder")]
    #[error("failed to crate posenet decoder delegate")]
    CreatePosenetDecoderDelegate,

    #[error("failed to get edgetpu device: no devices found")]
    GetEdgeTpuDevice,

    #[error("failed to get number of channels in Mat")]
    GetChannels(#[source] opencv::Error),

    #[error("failed to get Mat data")]
    GetMatData(#[source] opencv::Error),

    #[error("failed to convert num poses value to usize")]
    ConvertNumPosesToUSize,

    #[error("failed to construct delegate, got null pointer")]
    ConstructDelegate,

    #[error("failed to reshape offsets from {0:?} to {1:?}")]
    ReshapeOffsets(
        #[source] ndarray::ShapeError,
        (usize, usize, usize),
        [usize; 4],
    ),

    #[error("failed to reshape displacements_fwd from {0:?} to {1:?}")]
    ReshapeFwdDisplacements(
        #[source] ndarray::ShapeError,
        (usize, usize, usize, usize),
        [usize; 4],
    ),

    #[error("failed to reshape displacements_bwd from {0:?} to {1:?}")]
    ReshapeBwdDisplacements(
        #[source] ndarray::ShapeError,
        (usize, usize, usize, usize),
        [usize; 4],
    ),
}

pub(crate) fn check_null_mut<T>(ptr: *mut T) -> Option<*mut T> {
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}

///
pub(crate) fn check_null<T>(ptr: *const T) -> Option<*const T> {
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}

pub(crate) fn tflite_status_to_result(
    status: tflite_sys::TfLiteStatus,
    msg: &'static str,
) -> Result<(), Error> {
    use tflite_sys::TfLiteStatus;

    match status {
        TfLiteStatus::kTfLiteOk => Ok(()),
        TfLiteStatus::kTfLiteError => Err(Error::TfLite(msg)),
        TfLiteStatus::kTfLiteDelegateError => Err(Error::Delegate(msg)),
        TfLiteStatus::kTfLiteApplicationError => Err(Error::Application(msg)),
        _ => panic!("unknown error code: {:?}", status),
    }
}
