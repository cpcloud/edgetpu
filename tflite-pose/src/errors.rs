#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("failed to convert inference milliseconds to duration")]
    ConvertInferenceMillis(#[source] std::num::TryFromIntError),

    #[error("failed to get typed data from OpenCV Mat")]
    GetTypedData(#[source] opencv::Error),
}
