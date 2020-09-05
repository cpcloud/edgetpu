import enum
from enum import Enum, auto
from typing import Mapping, Tuple


@enum.unique
class Keypoint(Enum):
    Nose = auto()
    LeftEye = auto()
    RightEye = auto()
    LeftEar = auto()
    RightEar = auto()
    LeftShoulder = auto()
    RightShoulder = auto()
    LeftElbow = auto()
    RightElbow = auto()
    LeftWrist = auto()
    RightWrist = auto()
    LeftHip = auto()
    RightHip = auto()
    LeftKnee = auto()
    RightKnee = auto()
    LeftAnkle = auto()
    RightAnkle = auto()


IndexedKeypoint = {i: keypoint for i, keypoint in enumerate(Keypoint)}


NUM_KEYPOINTS = len(Keypoint)


X = int
Y = int


class PoseKeypoint:
    __slots__ = "keypoint", "point", "score"

    def __init__(self, keypoint: Keypoint, point: Tuple[X, Y], score: float) -> None:
        self.keypoint = keypoint
        self.point = point
        self.score = score

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.keypoint.name!r}, {self.point}, {self.score:.2f})"  # noqa: E501


class Pose:
    __slots__ = "keypoints", "score"

    def __init__(
        self, keypoints: Mapping[Keypoint, PoseKeypoint], score: float
    ) -> None:
        assert len(keypoints) == len(Keypoint)
        self.keypoints = keypoints
        self.score = score

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.keypoints}, {self.score:.2f})"
