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

    #[error("failed to get edgetpu device: no devices found")]
    GetEdgeTpuDevice,

    #[error("failed to find unallocated edgetpu device")]
    FindEdgeTpuDevice,

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

    #[error("failed to get pointer to interpreter for pipelined model runner")]
    GetInterpreter,

    #[error("failed to create PipelinedModelRunner")]
    CreatePipelinedModelRunner,

    #[error("failed to convert number of input tensors from i32 to usize")]
    GetNumOutputTensors(#[source] std::num::TryFromIntError),

    #[error("failed to pop pipelined model output tensors")]
    PopPipelinedModelOutputTensors,

    #[error("got null pointer when initializing input tensor")]
    AllocInputTensor,

    #[error("failed to get output interpreter")]
    GetOutputInterpreter,

    #[error("cannot construct pose engine from empty model paths")]
    ConstructPoseEngine,

    #[error("failed to canonicalize path: {1:?}")]
    CanonicalizePath(#[source] std::io::Error, std::path::PathBuf),

    #[error("failed to get tensor name")]
    GetTensorName(#[source] std::str::Utf8Error),

    #[error("failed to get tensor with name: {0}")]
    GetOutputTensorByName(String),

    #[error("failed to get segment stats")]
    GetSegmentStats,

    #[error("got zero segment stats")]
    EmptySegmentStats,

    #[error("cannot construct PipelinedModelRunner from empty interpreters list")]
    ConstructPipelineModelRunnerFromInterpreters,

    #[error("failed to get edgetpu device path")]
    GetEdgeTpuDevicePath(#[source] std::str::Utf8Error),

    #[error("failed to push tensors")]
    PushTensors,

    #[error("got empty interpreter pointers vec")]
    GetInterpreterPointers,

    #[error("got null interpreter vec pointer")]
    GetInterpreterVecPointer,

    #[error("got null pointer when constructing PipelineInputTensor")]
    GetPipelineInputTensor,

    #[error("got null pointer when constructing PipelineOutputTensor")]
    GetPipelineOutputTensor,

    #[error("failed to convert i64 to u64")]
    ConvertI64ToU64(#[source] std::num::TryFromIntError),

    #[error("queue sizes pointer was null")]
    GetQueueSizesPointer,
}

/// Check whether a pointer to mut T data is null.
pub(crate) fn check_null_mut<T>(ptr: *mut T) -> Option<*mut T> {
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}

/// Check whether a pointer to const T is null.
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
