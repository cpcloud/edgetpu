import numpy as np
from edgetpu.basic.basic_engine import BasicEngine

from posenet.core import NUM_KEYPOINTS, IndexedKeypoint, Pose, PoseKeypoint


class PoseEngine(BasicEngine):
    """Engine used for pose tasks."""

    def __init__(self, model_path, mirror=False):
        """Creates a PoseEngine with given model.

        Args:
          model_path: String, path to TF-Lite Flatbuffer file.
          mirror: Flip keypoints horizontally

        Raises:
          ValueError: An error occurred when model output is invalid.
        """
        BasicEngine.__init__(self, model_path)
        self._mirror = mirror

        self._input_tensor_shape = self.get_input_tensor_shape()
        if (
            self._input_tensor_shape.size != 4
            or self._input_tensor_shape[3] != 3
            or self._input_tensor_shape[0] != 1
        ):
            raise ValueError(
                (
                    "Image model should have input shape [1, height, width, 3]!"
                    " This model has {}.".format(self._input_tensor_shape)
                )
            )
        (
            _,
            self.image_height,
            self.image_width,
            self.image_depth,
        ) = self.get_input_tensor_shape()

        # The API returns all the output tensors flattened and concatenated. We
        # have to figure out the boundaries from the tensor shapes & sizes.
        offset = 0
        self._output_offsets = [0]
        for size in self.get_all_output_tensors_sizes():
            offset += size
            self._output_offsets.append(int(offset))

    def DetectPosesInImage(self, img):
        """Detects poses in a given image.

           For ideal results make sure the image fed to this function is close to the
           expected input size - it is the caller's responsibility to resize the
           image accordingly.

        Args:
          img: numpy array containing image
        """

        # Extend or crop the input to match the input shape of the network.
        if img.shape[0] < self.image_height or img.shape[1] < self.image_width:
            img = np.pad(
                img,
                [
                    [0, max(0, self.image_height - img.shape[0])],
                    [0, max(0, self.image_width - img.shape[1])],
                    [0, 0],
                ],
                mode="constant",
            )
        img = img[0 : self.image_height, 0 : self.image_width]
        assert img.shape == tuple(self._input_tensor_shape[1:])

        # Run the inference (API expects the data to be flattened)
        return self.ParseOutput(self.run_inference(img.flatten()))

    def ParseOutput(self, output):
        inference_time, output = output
        outputs = [
            output[i:j] for i, j in zip(self._output_offsets, self._output_offsets[1:])
        ]

        keypoints = outputs[0].reshape(-1, NUM_KEYPOINTS, 2)
        keypoint_scores = outputs[1].reshape(-1, NUM_KEYPOINTS)
        pose_scores = outputs[2]
        nposes = int(outputs[3][0])
        assert nposes < outputs[0].shape[0]

        # Convert the poses to a friendlier format of keypoints with associated
        # scores.
        poses = []
        for pose_i in range(nposes):
            keypoint_dict = {}
            for point_i, (y, x) in enumerate(keypoints[pose_i]):
                keypoint = PoseKeypoint(
                    IndexedKeypoint[point_i], (x, y), keypoint_scores[pose_i, point_i]
                )
                if self._mirror:
                    keypoint.yx[1] = self.image_width - keypoint.yx[1]
                keypoint_dict[IndexedKeypoint[point_i]] = keypoint
            poses.append(Pose(keypoint_dict, pose_scores[pose_i]))

        return poses, inference_time
