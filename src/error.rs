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

    #[error("failed to get output tensor: got null pointer instead")]
    GetOutputTensor,

    #[error("expected {0} output tensors, got {1}")]
    GetExpectedNumOutputs(usize, usize),

    #[error("failed to convert row count i32 to usize")]
    ConvertRowsToUsize(#[source] std::num::TryFromIntError),

    #[error("failed to get value of Mat::elemSize1")]
    GetElemSize1(#[source] opencv::Error),

    #[error("failed to get value of Mat::step1")]
    GetStep1(#[source] opencv::Error),

    #[cfg(feature = "posenet_decoder")]
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

    #[error("failed to construct delegate, got null pointer")]
    ConstructDelegate,

    #[error("failed to convert usize index to i32 index")]
    ConvertUSizeToI32Index(#[source] std::num::TryFromIntError),

    #[error("failed to convert dim i32 to usize")]
    ConvertDimI32ToUSize(#[source] std::num::TryFromIntError),

    #[error("failed to convert dim i32 to usize")]
    GetNumDims(#[source] std::num::TryFromIntError),

    #[error("got null pointer when constructing Tensor")]
    CreateTensor,

    #[error("dimension index {0} is out of bounds for tensor with dimensions {1}")]
    GetDim(usize, usize),

    #[error("failed to convert value to f32")]
    ConvertToF32,

    #[error("failed to convert value to f64")]
    ConvertToF64,

    #[error("failed to convert value to usize")]
    ConvertToUSize,

    #[error("failed to construct ArrayView3 from Mat")]
    ConstructNDArrayFromMat(#[source] ndarray::ShapeError),

    #[error("failed to get ndarray as contiguous slice")]
    GetNDArrayAsSlice,

    #[error("failed to construct NotNan from f32: {1}")]
    ConstructNotNan(#[source] ordered_float::FloatIsNan, f32),

    #[error("tensor type is not valid for converting to slice: {0:?}")]
    GetTensorSlice(tflite_sys::TfLiteType),

    #[cfg(feature = "gui")]
    #[error("failed to draw line")]
    DrawLine(#[source] opencv::Error),

    #[cfg(feature = "gui")]
    #[error("failed to draw circle")]
    DrawCircle(#[source] opencv::Error),

    #[cfg(feature = "gui")]
    #[error("failed to draw text")]
    PutText(#[source] opencv::Error),

    #[cfg(feature = "gui")]
    #[error("failed to show image")]
    ImShow(#[source] opencv::Error),

    #[cfg(feature = "gui")]
    #[error("failed to convert Point2f {0:?} to Point2i")]
    ConvertPoint2fToPoint2i(opencv::core::Point2f),

    #[error("failed to send pipelined output tensors to channel")]
    SendPipelinedOutputTensors(
        #[error] std::sync::mpsc::SendError<Vec<crate::coral::PipelineTensor>>,
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
