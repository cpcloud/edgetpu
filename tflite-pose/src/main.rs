#![feature(variant_count)]

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("foo")]
    InputDimensions,

    #[error("bar")]
    FirstDimensionSize,

    #[error("baz")]
    DepthSize,
}

#[derive(Debug, Copy, Clone)]
enum KeypointKind {
    Nose,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
}

#[derive(Debug, Copy, Clone)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Debug, Copy, Clone)]
struct Keypoint {
    kind: KeypointKind,
    point: Point,
    score: f64,
}

type Keypoints = [Keypoint; std::mem::variant_count::<KeypointKind>()];

struct Pose<const N: usize> {
    keypoints: [Keypoint; N],
}

fn main() {

}
