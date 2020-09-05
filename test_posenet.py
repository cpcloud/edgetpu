#!/usr/bin/env python

# flake8: E203

"""Doing some pose stuff in Python."""

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import click
import cv2
import numpy as np
from edgetpu.basic.basic_engine import BasicEngine

from posenet.core import NUM_KEYPOINTS, Keypoint
from posenet.decode_multi import decode_multiple_poses
from posenet.pose_engine import PoseEngine

OUTPUT_STRIDE = 16


def extract_outputs(outputs, engine):
    """Extract heatmaps, offsets, and displacement vectors"""

    # Input image and heatmap dimensions
    _, img_h, img_w, _ = engine.get_input_tensor_shape()
    height = int(1 + (img_h - 1) / OUTPUT_STRIDE)
    width = int(1 + (img_w - 1) / OUTPUT_STRIDE)

    # Reshape output tensors
    out_sz = engine.get_all_output_tensors_sizes()

    # Heatmaps
    ofs = np.uint64(0)
    heatmaps = outputs[ofs : ofs + out_sz[0]].reshape(height, width, NUM_KEYPOINTS)
    ofs += out_sz[0]

    # Offsets - [height, width, 2, 17]
    offsets = outputs[ofs : ofs + out_sz[1]].reshape(height, width, NUM_KEYPOINTS * 2)
    ofs += out_sz[1]

    # Displacement vectors (FWD, BWD): size [height, width, 4, 16], columns
    # [fwd_i, fwd_j, bwd_i, bwd_j]
    raw_dsp = outputs[ofs : ofs + out_sz[2]].reshape(height, width, 4, -1)
    fwd = raw_dsp[:, :, 0:2, :]
    bwd = raw_dsp[:, :, 2:4, :]

    return {
        "heatmaps": 1.0 / (1.0 + np.exp(-heatmaps)),
        "offsets": offsets,
        "displacements_fwd": fwd,
        "displacements_bwd": bwd,
    }


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


def pose_engine_poses(engine: BasicEngine, frame) -> Tuple[float, List[Pose]]:
    poses, inference_time = engine.DetectPosesInImage(frame)
    return inference_time, poses


def basic_engine_poses(engine, frame) -> Tuple[float, List[Pose]]:
    inference_time, output = engine.run_inference(frame.ravel())
    # Decode pose
    out = extract_outputs(output, engine)
    pose_scores, all_keypoint_scores, all_keypoint_coords = decode_multiple_poses(
        out["heatmaps"],
        out["offsets"],
        out["displacements_fwd"],
        out["displacements_bwd"],
        OUTPUT_STRIDE,
        max_pose_detections=10,
        score_threshold=0.5,
        nms_radius=20,
        min_pose_score=0.1,
    )
    poses = [
        Pose(
            {
                keypoint: PoseKeypoint(keypoint, (int(x), int(y)), keypoint_score)
                for keypoint, keypoint_score, (y, x) in zip(
                    Keypoint, keypoint_scores, keypoint_coords
                )
            },
            pose_score,
        )
        for pose_score, keypoint_scores, keypoint_coords in zip(
            pose_scores, all_keypoint_scores, all_keypoint_coords
        )
    ]
    return inference_time, poses


EDGES = (
    (Keypoint.Nose, Keypoint.LeftEye),
    (Keypoint.Nose, Keypoint.RightEye),
    (Keypoint.Nose, Keypoint.LeftEar),
    (Keypoint.Nose, Keypoint.RightEar),
    (Keypoint.LeftEar, Keypoint.LeftEye),
    (Keypoint.RightEar, Keypoint.RightEye),
    (Keypoint.LeftEye, Keypoint.RightEye),
    (Keypoint.LeftShoulder, Keypoint.RightShoulder),
    (Keypoint.LeftShoulder, Keypoint.LeftElbow),
    (Keypoint.LeftShoulder, Keypoint.LeftHip),
    (Keypoint.RightShoulder, Keypoint.RightElbow),
    (Keypoint.RightShoulder, Keypoint.RightHip),
    (Keypoint.LeftElbow, Keypoint.LeftWrist),
    (Keypoint.RightElbow, Keypoint.RightWrist),
    (Keypoint.LeftHip, Keypoint.RightHip),
    (Keypoint.LeftHip, Keypoint.LeftKnee),
    (Keypoint.RightHip, Keypoint.RightKnee),
    (Keypoint.LeftKnee, Keypoint.LeftAnkle),
    (Keypoint.RightKnee, Keypoint.RightAnkle),
)


def draw_skeletons(poses: Sequence[Pose], frame, threshold: float) -> None:
    for pose in poses:
        xys = {}
        for _, pose_keypoint in pose.keypoints.items():
            if pose_keypoint.score >= threshold:
                point = tuple(map(int, pose_keypoint.point))
                xys[pose_keypoint.keypoint] = point
                cv2.circle(frame, point, 6, (0, 255, 0), -1)

        for a, b in EDGES:
            if a in xys and b in xys:
                cv2.line(frame, xys[a], xys[b], (0, 255, 255), 2)
    cv2.imshow("Pose", frame)


@click.group()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-w", "--model-width", type=int, default=481)
@click.option("-H", "--model-height", type=int, default=353)
@click.option("--basic/--no-basic")
@click.option("-t", "threshold", type=float, default=0.0)
@click.option("-d", "--edgetpu-device", type=click.Path(), default=None)
@click.pass_context
def cli(
    ctx,
    model_path: str,
    model_width: int,
    model_height: int,
    basic: bool,
    threshold: float,
    edgetpu_device: Optional[str],
):
    ctx.ensure_object(dict)

    engine = (
        BasicEngine(model_path, edgetpu_device)
        if basic
        else PoseEngine(model_path, edgetpu_device)
    )
    func = basic_engine_poses if basic else pose_engine_poses
    ctx.obj["engine"] = engine
    ctx.obj["func"] = func
    ctx.obj["threshold"] = threshold
    ctx.obj["model_dimensions"] = model_width, model_height
    ctx.obj["output_kind"] = "no_decoder" if basic else "decoder"


@cli.command()
@click.option(
    "-i",
    "--image",
    type=click.Path(exists=True),
    default=str(Path(__file__).parent.parent / "couple.jpg"),
)
@click.pass_context
def image(ctx, image: str):
    engine = ctx.obj["engine"]
    func = ctx.obj["func"]
    model_dimensions = ctx.obj["model_dimensions"]
    threshold = ctx.obj["threshold"]
    frame = cv2.imread(image)
    frame = cv2.resize(frame, model_dimensions)
    inference_time, poses = func(engine, frame)
    draw_skeletons(poses, frame, threshold)
    output_kind = ctx.obj["output_kind"]
    cv2.imwrite(f"./{output_kind}.jpg", frame)
    cv2.imshow("Pose", frame)
    if cv2.waitKey(0) > 0:
        return 0


@cli.command()
@click.option(
    "-c", "--capture-device", type=click.Path(exists=True), default="/dev/video2"
)
@click.pass_context
def video(
    ctx,
    model_path: str,
    image: str,
    model_width: int,
    model_height: int,
    basic: bool,
    threshold: float,
):
    cap = cv2.VideoCapture("/dev/video2")
    engine = ctx.obj["engine"]
    func = ctx.obj["func"]
    model_dimensions = ctx.obj["model_dimensions"]
    threshold = ctx.obj["threshold"]
    while True:
        frame = cv2.imread(image)
        success, frame = cap.read()
        if not success:
            return 1
        frame = cv2.resize(frame, model_dimensions)
        inference_time, poses = func(engine, frame)
        draw_skeletons(poses, frame, threshold)
        cv2.imshow("Pose", frame)
        if cv2.waitKey(0) > 0:
            return 0


if __name__ == "__main__":
    cli(obj={})
