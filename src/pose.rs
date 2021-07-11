use crate::error::Error;
use num_traits::ToPrimitive;

#[derive(Debug, Copy, Clone, num_derive::FromPrimitive, num_derive::ToPrimitive)]
pub(crate) enum KeypointKind {
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

impl KeypointKind {
    pub(crate) fn idx(self) -> Result<usize, Error> {
        self.to_usize().ok_or(Error::KeypointVariantToUSize(self))
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct Keypoint {
    pub(crate) kind: Option<KeypointKind>,
    pub(crate) point: opencv::core::Point2f,
    pub(crate) score: f32,
}

pub(crate) const NUM_KEYPOINTS: usize = std::mem::variant_count::<KeypointKind>();
pub(crate) type Keypoints = [Keypoint; NUM_KEYPOINTS];

#[derive(Debug, Copy, Clone)]
pub(crate) struct Pose {
    pub(crate) keypoints: Keypoints,
    pub(crate) score: f32,
}

pub(crate) mod constants {
    use crate::pose::KeypointKind::{self, *};

    pub(crate) const LOCAL_MAXIMUM_RADIUS: usize = 1;
    pub(crate) const EDGE_LIST: [(KeypointKind, KeypointKind); 32] = [
        // forward edges
        (Nose, LeftEye),
        (LeftEye, LeftEar),
        (Nose, RightEye),
        (RightEye, RightEar),
        (Nose, LeftShoulder),
        (LeftShoulder, LeftElbow),
        (LeftElbow, LeftWrist),
        (LeftShoulder, LeftHip),
        (LeftHip, LeftKnee),
        (LeftKnee, LeftAnkle),
        (Nose, RightShoulder),
        (RightShoulder, RightElbow),
        (RightElbow, RightWrist),
        (RightShoulder, RightHip),
        (RightHip, RightKnee),
        (RightKnee, RightAnkle),
        // backward edges
        (LeftEye, Nose),
        (LeftEar, LeftEye),
        (RightEye, Nose),
        (RightEar, RightEye),
        (LeftShoulder, Nose),
        (LeftElbow, LeftShoulder),
        (LeftWrist, LeftElbow),
        (LeftHip, LeftShoulder),
        (LeftKnee, LeftHip),
        (LeftAnkle, LeftKnee),
        (RightShoulder, Nose),
        (RightElbow, RightShoulder),
        (RightWrist, RightElbow),
        (RightHip, RightShoulder),
        (RightKnee, RightHip),
        (RightAnkle, RightKnee),
    ];

    #[cfg(feature = "gui")]
    pub(crate) const KEYPOINT_EDGES: [(KeypointKind, KeypointKind); 19] = [
        (Nose, LeftEye),
        (Nose, RightEye),
        (Nose, LeftEar),
        (Nose, RightEar),
        (LeftEar, LeftEye),
        (RightEar, RightEye),
        (LeftEye, RightEye),
        (LeftShoulder, RightShoulder),
        (LeftShoulder, LeftElbow),
        (LeftShoulder, LeftHip),
        (RightShoulder, RightElbow),
        (RightShoulder, RightHip),
        (LeftElbow, LeftWrist),
        (RightElbow, RightWrist),
        (LeftHip, RightHip),
        (LeftHip, LeftKnee),
        (RightHip, RightKnee),
        (LeftKnee, LeftAnkle),
        (RightKnee, RightAnkle),
    ];
}
