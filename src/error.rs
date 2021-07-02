use crate::tflite_sys;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("failed to get output tensor: got null pointer instead")]
    GetOutputTensor,

    #[error("failed to get output tensor from cxx")]
    GetOutputTensorFromCxx(#[source] cxx::Exception),

    #[error("failed to get output tensor count from cxx")]
    GetOutputTensorCount(#[source] cxx::Exception),

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

    #[error("failed to convert keypoint variant to usize: {0:?}")]
    KeypointVariantToUSize(crate::pose::KeypointKind),

    #[error("failed to get Mat data")]
    GetMatData(#[source] opencv::Error),

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

    #[error("cannot construct PipelinedModelRunner from empty interpreters list")]
    ConstructPipelineModelRunnerFromInterpreters,

    #[error("got empty interpreter pointers vec")]
    GetInterpreterPointers,

    #[error("failed to get model path as &str: {0:?}")]
    GetModelPathAsStr(std::path::PathBuf),

    #[error("got null pointer for TfLiteTensor when computing dimension size")]
    GetTensorForDim,

    #[error("got null pointer for TfLiteTensor when computing number of dimensions")]
    GetTensorForNumDims,

    #[error("failed to push input tensors")]
    PushInputTensors(#[source] cxx::Exception),

    #[error("failed to pop output tensors")]
    PopOutputTensors(#[source] cxx::Exception),

    #[error("failed to make model")]
    MakeModel(#[source] cxx::Exception),

    #[error("failed to make interpreter from model")]
    MakeInterpreterFromModel(#[source] cxx::Exception),

    #[error("failed to get edgetpu device infos")]
    GetAllDeviceInfos(#[source] cxx::Exception),

    #[error("failed to make edge tpu context")]
    MakeEdgeTpuContext(#[source] cxx::Exception),

    #[error("failed to construct input tensor")]
    MakeInputTensor(#[source] cxx::Exception),

    #[error("failed to get output queue size")]
    GetQueueSizes(#[source] cxx::Exception),

    #[error("failed to set output queue size")]
    SetOutputQueueSize(#[source] cxx::Exception),

    #[error("failed set input queue size")]
    SetInputQueueSize(#[source] cxx::Exception),

    #[error("failed to generate pipelined model runner")]
    MakePipelinedModelRunner(#[source] cxx::Exception),
}

/// Check whether a pointer to const T is null.
pub(crate) fn check_null<T>(ptr: *const T) -> Option<*const T> {
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}
