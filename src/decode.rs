use crate::{
    error::Error,
    pose::{self, Keypoint, KeypointKind, Pose},
    tflite::{self, TensorElement, TypedTensor},
};
use ndarray::{
    array, s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis,
    ShapeBuilder,
};
use num_traits::cast::{FromPrimitive, ToPrimitive};

pub(crate) trait Decoder {
    /// Return the number of expected_output_tensors the decoder expects to operate on.
    fn expected_output_tensors(&self) -> usize;

    /// Decode poses into a Vec of Pose.
    fn decode<I, O>(
        &self,
        interp: &mut tflite::Interpreter<I, O>,
        dims: (u16, u16),
    ) -> Result<Vec<pose::Pose>, Error>
    where
        O: TensorElement;

    /// Validate that the model has the expected number of output tensors.
    fn validate_output_tensor_count(&self, output_tensor_count: usize) -> Result<(), Error> {
        let expected_output_tensors = self.expected_output_tensors();
        if output_tensor_count != expected_output_tensors {
            Err(Error::GetExpectedNumOutputs(
                expected_output_tensors,
                output_tensor_count,
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "posenet_decoder")]
#[derive(Debug, Clone, Copy, Default, structopt::StructOpt)]
pub(crate) struct PosenetDecoder {}

#[cfg(feature = "posenet_decoder")]
impl Decoder for PosenetDecoder {
    fn expected_output_tensors(&self) -> usize {
        4
    }

    fn decode<I, O>(
        &self,
        interp: &mut tflite::Interpreter<I, O>,
        (_width, _height): (u16, u16),
    ) -> Result<Vec<pose::Pose>, Error>
    where
        O: TensorElement,
    {
        // construct the output tensors
        let pose_keypoints = interp.get_output_tensor(0)?;
        let pose_keypoints = pose_keypoints.as_ndarray(
            *(
                pose_keypoints.dim(1),
                pose_keypoints.dim(2),
                pose_keypoints.dim(3),
            )
                .into_shape()
                .raw_dim(),
        )?;
        let keypoint_scores = interp.get_output_tensor(1)?;
        let keypoint_scores = keypoint_scores.as_ndarray(
            *(keypoint_scores.dim(1), keypoint_scores.dim(2))
                .into_shape()
                .raw_dim(),
        )?;
        let pose_scores = interp.get_output_tensor(2)?;
        let pose_scores = pose_scores.as_ndarray(*pose_scores.dim(1).into_shape().raw_dim())?;
        let nposes = interp.get_output_tensor(3)?.as_slice()[0] as usize;

        // move poses into a more useful structure
        let mut poses = Vec::with_capacity(nposes);

        for (pose_i, keypoint) in pose_keypoints.axis_iter(ndarray::Axis(0)).enumerate() {
            let mut keypoint_map: pose::Keypoints = Default::default();

            for (point_i, point) in keypoint.axis_iter(ndarray::Axis(0)).enumerate() {
                keypoint_map[point_i] = Keypoint {
                    kind: Some(
                        KeypointKind::from_usize(point_i)
                            .ok_or(Error::ConvertUSizeToKeypointKind(point_i))?,
                    ),
                    point: opencv::core::Point2f::new(point[1], point[0]),
                    score: keypoint_scores[(pose_i, point_i)],
                };
            }

            poses.push(Pose {
                keypoints: keypoint_map,
                score: pose_scores[pose_i],
            });
        }

        Ok(poses)
    }
}

#[derive(Debug, Clone, Copy, structopt::StructOpt)]
pub(crate) struct HandRolledDecoder {
    #[structopt(short, long, default_value = "16")]
    pub(crate) output_stride: u8,
    #[structopt(short, long, default_value = "100")]
    pub(crate) max_pose_detections: usize,
    #[structopt(short, long, default_value = "0.2")]
    pub(crate) score_threshold: f32,
    #[structopt(short, long, default_value = "20")]
    pub(crate) nms_radius: usize,
}

struct Part {
    score: f32,
    keypoint_id: usize,
    x: usize,
    y: usize,
}

impl HandRolledDecoder {
    fn traverse_target_to_keypoint(
        &self,
        edge_id: usize,
        source_keypoint: ArrayView1<f32>,
        target_keypoint_id: usize,
        scores: ArrayView3<f32>,
        offsets: ArrayView4<usize>,
        displacements: ArrayView4<usize>,
    ) -> (f32, Array1<f32>) {
        let (height, width, _) = scores.dim();
        let rounded_source_keypoints =
            (&source_keypoint / f32::from(self.output_stride)).mapv(|v| v.round());
        let source_keypoint_indices = array![
            rounded_source_keypoints[0]
                .clamp(0.0_f32, height.to_f32().unwrap() - 1.0_f32)
                .to_usize()
                .unwrap(),
            rounded_source_keypoints[1]
                .clamp(0.0_f32, width.to_f32().unwrap() - 1.0_f32)
                .to_usize()
                .unwrap(),
        ];
        let displaced_points = &source_keypoint
            + displacements
                .slice(s![
                    source_keypoint_indices[0],
                    source_keypoint_indices[1],
                    edge_id,
                    ..,
                ])
                .mapv(|v| v.to_f32().unwrap());
        let displaced_point_indices = array![
            displaced_points[0]
                .clamp(0.0_f32, height.to_f32().unwrap() - 1.0_f32)
                .to_usize()
                .unwrap(),
            displaced_points[1]
                .clamp(0.0_f32, width.to_f32().unwrap() - 1.0_f32)
                .to_usize()
                .unwrap(),
        ];
        let score = scores[(
            displaced_point_indices[0],
            displaced_point_indices[1],
            target_keypoint_id,
        )];
        let image_coord = &displaced_point_indices
            * usize::from(self.output_stride)
            * offsets.slice(s![
                displaced_point_indices[0],
                displaced_point_indices[1],
                target_keypoint_id,
                ..
            ]);
        (score, image_coord.mapv(|v| v.to_f32().unwrap()))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_pose(
        &self,
        root_score: f32,
        root_id: usize,
        root_image_coord: Array1<f32>,
        scores: ArrayView3<f32>,
        offsets: ArrayView4<usize>,
        displacements_fwd: ArrayView4<usize>,
        displacements_bwd: ArrayView4<usize>,
    ) -> (Array1<f32>, Array2<f32>) {
        let (_, _, num_parts) = scores.dim();

        let mut instance_keypoint_scores = Array1::zeros(num_parts);
        let mut instance_keypoint_coords = Array2::zeros((num_parts, 2));

        instance_keypoint_scores[root_id] = root_score;
        instance_keypoint_coords
            .slice_mut(s![root_id, ..])
            .assign(&root_image_coord);

        let mut apply = |edge_i: usize,
                         source_keypoint_id: &KeypointKind,
                         target_keypoint_id: &KeypointKind,
                         displacements: ArrayView4<usize>| {
            let source_keypoint_id = source_keypoint_id.to_usize().unwrap();
            let target_keypoint_id = target_keypoint_id.to_usize().unwrap();
            if instance_keypoint_scores[source_keypoint_id] > 0.0
                && instance_keypoint_scores[target_keypoint_id] == 0.0
            {
                let (score, coords) = self.traverse_target_to_keypoint(
                    edge_i,
                    instance_keypoint_coords.slice(s![source_keypoint_id, ..]),
                    target_keypoint_id,
                    scores,
                    offsets,
                    displacements,
                );
                instance_keypoint_scores[target_keypoint_id] = score;
                instance_keypoint_coords
                    .slice_mut(s![target_keypoint_id, ..])
                    .assign(&coords);
            }
        };

        pose::constants::POSE_CHAIN
            .iter()
            .rev()
            .enumerate()
            .for_each(|(edge_i, (target_keypoint_id, source_keypoint_id))| {
                apply(
                    edge_i,
                    source_keypoint_id,
                    target_keypoint_id,
                    displacements_bwd,
                )
            });

        pose::constants::POSE_CHAIN.iter().enumerate().for_each(
            move |(edge_i, (source_keypoint_id, target_keypoint_id))| {
                apply(
                    edge_i,
                    source_keypoint_id,
                    target_keypoint_id,
                    displacements_fwd,
                )
            },
        );

        (instance_keypoint_scores, instance_keypoint_coords)
    }

    fn within_nms_radius_fast(&self, pose_coords: ArrayView2<f32>, point: ArrayView1<f32>) -> bool {
        let squared_nms_radius = self.nms_radius.pow(2).to_f32().unwrap();
        pose_coords.dim().0 != 0
            && (&pose_coords - &point)
                .fold_axis(Axis(1), 0.0, |&total, &value| total + value.powi(2))
                .into_iter()
                .any(|v| v <= squared_nms_radius)
    }

    fn get_instance_score_fast(
        &self,
        // k x m x n
        exist_pose_coords: ArrayView3<f32>,
        keypoint_scores: ArrayView1<f32>,
        // m x n
        keypoint_coords: ArrayView2<f32>,
    ) -> f32 {
        let squared_nms_radius = self.nms_radius.pow(2).to_f32().unwrap();
        let denominator = keypoint_scores.len().to_f32().unwrap();
        let (first_dim, ..) = exist_pose_coords.dim();

        let sum = if first_dim != 0 {
            let mask = (&exist_pose_coords - &keypoint_coords)
                .fold_axis(Axis(2), 0.0, |&total, &value| total + value.powi(2))
                .fold_axis(Axis(0), true, |&previous, &value| {
                    previous && value > squared_nms_radius
                });
            keypoint_scores
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if mask[i] { Some(v) } else { None })
                .sum()
        } else {
            keypoint_scores.sum()
        };
        sum / denominator
    }

    fn score_is_max_in_local_window(
        &self,
        score: f32,
        hmy: usize,
        hmx: usize,
        local_maximum_radius: usize,
        scores: ArrayView2<f32>,
    ) -> bool {
        let (height, width) = scores.dim();
        let y_start = 0.max(hmy - local_maximum_radius);
        let y_end = height.min(hmy + local_maximum_radius + 1);
        let x_start = 0.max(hmx - local_maximum_radius);
        let x_end = width.min(hmx + local_maximum_radius + 1);
        scores
            .slice(s![y_start..y_end, x_start..x_end])
            .fold(true, |previous, &value| previous && value <= score)
    }

    fn build_part_with_score(
        &self,
        local_maximum_radius: usize,
        scores: ArrayView3<f32>,
    ) -> Vec<Part> {
        scores
            .indexed_iter()
            .filter_map(|((hmy, hmx, keypoint_id), &score)| {
                if score >= self.score_threshold
                    && self.score_is_max_in_local_window(
                        score,
                        hmy,
                        hmx,
                        local_maximum_radius,
                        scores.index_axis(Axis(2), keypoint_id),
                    )
                {
                    Some(Part {
                        score,
                        keypoint_id,
                        x: hmx,
                        y: hmy,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    #[allow(clippy::type_complexity)]
    fn decode_multiple_poses(
        &self,
        scores: Array3<f32>,
        offsets: ArrayView3<f32>,
        displacements_fwd: ArrayView4<f32>,
        displacements_bwd: ArrayView4<f32>,
    ) -> Result<(Array1<f32>, Array2<f32>, Array3<f32>), Error> {
        let mut pose_scores = Array1::zeros(self.max_pose_detections);
        let mut pose_keypoint_scores =
            Array2::zeros((self.max_pose_detections, pose::NUM_KEYPOINTS));
        let mut pose_keypoint_coords =
            Array3::zeros((self.max_pose_detections, pose::NUM_KEYPOINTS, 2));

        let mut scored_parts =
            self.build_part_with_score(pose::constants::LOCAL_MAXIMUM_RADIUS, scores.view());
        scored_parts
            .sort_by_key(|part| ordered_float::NotNan::new(part.score).expect("value is NaN"));

        let (height, width, _) = scores.dim();

        let new_shape = [height, width, 2, offsets.len() / (height * width * 2)];

        const TRANSPOSE_AXES: [usize; 4] = [0, 1, 3, 2];

        let new_offsets = offsets
            .into_shape(new_shape)
            .map_err(|e| Error::ReshapeOffsets(e, offsets.dim(), new_shape))?
            .permuted_axes(TRANSPOSE_AXES)
            .map(|&v| v.to_usize().unwrap());
        let new_displacments_fwd = displacements_fwd
            .into_shape(new_shape)
            .map_err(|e| Error::ReshapeFwdDisplacements(e, displacements_fwd.dim(), new_shape))?
            .permuted_axes(TRANSPOSE_AXES)
            .map(|&v| v.to_usize().unwrap());
        let new_displacments_bwd = displacements_bwd
            .into_shape(new_shape)
            .map_err(|e| Error::ReshapeBwdDisplacements(e, displacements_bwd.dim(), new_shape))?
            .permuted_axes(TRANSPOSE_AXES)
            .map(|&v| v.to_usize().unwrap());

        for (
            pose_count,
            Part {
                keypoint_id,
                score,
                x,
                y,
            },
        ) in scored_parts.into_iter().enumerate()
        {
            let coord = array![y, x];
            let root_image_coord = (coord
                * usize::from(self.output_stride)
                * new_offsets.slice(s![y, x, keypoint_id, ..]))
            .map(|v| v.to_f32().unwrap());
            if !self.within_nms_radius_fast(
                pose_keypoint_coords.slice(s![..pose_count, keypoint_id, ..]),
                root_image_coord.view(),
            ) {
                let (keypoint_scores, keypoint_coords) = self.decode_pose(
                    score,
                    keypoint_id,
                    root_image_coord,
                    scores.view(),
                    new_offsets.view(),
                    new_displacments_fwd.view(),
                    new_displacments_bwd.view(),
                );
                pose_scores[pose_count] = self.get_instance_score_fast(
                    pose_keypoint_coords.slice(s![..pose_count, .., ..]),
                    keypoint_scores.view(),
                    keypoint_coords.view(),
                );
                pose_keypoint_scores
                    .slice_mut(s![pose_count, ..])
                    .assign(&keypoint_scores);
                pose_keypoint_coords
                    .slice_mut(s![pose_count, .., ..])
                    .assign(&keypoint_coords);
            }
        }

        Ok((pose_scores, pose_keypoint_scores, pose_keypoint_coords))
    }
}

impl Decoder for HandRolledDecoder {
    fn expected_output_tensors(&self) -> usize {
        3
    }

    fn decode<I, O>(
        &self,
        interp: &mut tflite::Interpreter<I, O>,
        (frame_width, frame_height): (u16, u16),
    ) -> Result<Vec<pose::Pose>, Error>
    where
        O: TensorElement,
    {
        let output_stride = u16::from(self.output_stride);
        let height = usize::from(1 + (frame_height - 1) / output_stride);
        let width = usize::from(1 + (frame_width - 1) / output_stride);

        let heatmaps = interp.get_output_tensor(0)?;
        let heatmaps = heatmaps
            .as_ndarray(*(height, width, pose::NUM_KEYPOINTS).into_shape().raw_dim())?
            .mapv(|v: f32| 1.0_f32 / (1.0_f32 + (-v).exp()));

        let offsets = interp.get_output_tensor(1)?;
        let offsets = offsets.as_ndarray(
            *(height, width, 2 * pose::NUM_KEYPOINTS)
                .into_shape()
                .raw_dim(),
        )?;

        let raw_dsp = interp.get_output_tensor(2)?;
        let raw_dsp = raw_dsp.as_ndarray(
            *(height, width, 4, usize::from(self.output_stride))
                .into_shape()
                .raw_dim(),
        )?;

        let fwd = raw_dsp.slice(s![.., .., ..2, ..]);
        let bwd = raw_dsp.slice(s![.., .., 2.., ..]);

        let (pose_scores, keypoint_scores, keypoints) =
            self.decode_multiple_poses(heatmaps, offsets, fwd, bwd)?;

        let (nposes, nkeypoints, _) = keypoints.dim();
        let mut poses = Vec::with_capacity(nposes);

        for pose_i in 0..nposes {
            let mut keypoint_map: pose::Keypoints = Default::default();

            for point_i in 0..nkeypoints {
                let point = keypoints.slice(s![pose_i, point_i, ..]);
                keypoint_map[point_i] = Keypoint {
                    kind: Some(
                        KeypointKind::from_usize(point_i)
                            .ok_or(Error::ConvertUSizeToKeypointKind(point_i))?,
                    ),
                    point: opencv::core::Point2f::new(point[1], point[0]),
                    score: keypoint_scores[(pose_i, point_i)],
                };
            }

            poses.push(Pose {
                keypoints: keypoint_map,
                score: pose_scores[pose_i],
            });
        }

        Ok(poses)
    }
}

#[derive(Debug, structopt::StructOpt)]
pub(crate) enum Decode {
    /// Decode using the builtin PosenetDecoderOp
    #[cfg(feature = "posenet_decoder")]
    Posenet(PosenetDecoder),
    /// Decode using a hand rolled decoder
    HandRolled(HandRolledDecoder),
}

#[cfg(feature = "posenet_decoder")]
impl Default for Decode {
    fn default() -> Self {
        Self::Posenet(PosenetDecoder {})
    }
}

impl Decoder for Decode {
    fn expected_output_tensors(&self) -> usize {
        match self {
            #[cfg(feature = "posenet_decoder")]
            Self::Posenet(d) => d.expected_output_tensors(),
            Self::HandRolled(d) => d.expected_output_tensors(),
        }
    }

    fn decode<I, O>(
        &self,
        interp: &mut tflite::Interpreter<I, O>,
        dims: (u16, u16),
    ) -> Result<Vec<pose::Pose>, Error>
    where
        O: TensorElement,
    {
        match self {
            #[cfg(feature = "posenet_decoder")]
            Self::Posenet(d) => d.decode(interp, dims),
            Self::HandRolled(d) => d.decode(interp, dims),
        }
    }
}
