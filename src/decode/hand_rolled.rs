use crate::{
    error::Error,
    pose::{self, KeypointKind},
    tflite,
};
use ndarray::{
    array, s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis,
};
use num_traits::cast::ToPrimitive;

#[derive(Debug, Clone, Copy, structopt::StructOpt)]
pub(crate) struct Decoder {
    #[structopt(short, long, default_value = "16")]
    pub(crate) output_stride: u8,
    #[structopt(short = "-M", long, default_value = "100")]
    pub(crate) max_pose_detections: usize,
    #[structopt(short, long, default_value = "0.5")]
    pub(crate) score_threshold: f32,
    #[structopt(short, long, default_value = "20")]
    pub(crate) nms_radius: usize,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            output_stride: 16,
            max_pose_detections: 100,
            score_threshold: 0.5,
            nms_radius: 20,
        }
    }
}

struct Part {
    score: f32,
    keypoint_id: usize,
    x: usize,
    y: usize,
}

impl Decoder {
    fn traverse_target_to_keypoint(
        &self,
        edge_id: usize,
        source_keypoint: ArrayView1<f32>,
        target_keypoint_id: usize,
        scores: ArrayView3<f32>,
        offsets: ArrayView4<usize>,
        displacements: ArrayView4<usize>,
    ) -> Result<(f32, Array1<f32>), Error> {
        let (height, width, _) = scores.dim();
        let rounded_source_keypoints =
            source_keypoint.map(|v| (v / f32::from(self.output_stride)).round());
        let height_float = height.to_f32().ok_or(Error::ConvertToF32)?;
        let width_float = width.to_f32().ok_or(Error::ConvertToF32)?;
        let source_keypoint_indices = array![
            rounded_source_keypoints[0]
                .clamp(0.0, height_float - 1.0)
                .to_usize()
                .ok_or(Error::ConvertToUSize)?,
            rounded_source_keypoints[1]
                .clamp(0.0, width_float - 1.0)
                .to_usize()
                .ok_or(Error::ConvertToUSize)?
        ];
        let displaced_points = &source_keypoint
            + displacements
                .slice(s![
                    source_keypoint_indices[0],
                    source_keypoint_indices[1],
                    edge_id,
                    ..,
                ])
                .mapv(|v| v.to_f32().expect("failed to convert usize to f32"));
        let displaced_point_indices = array![
            displaced_points[0]
                .clamp(0.0, height_float - 1.0)
                .to_usize()
                .expect("failed to convert f32 to usize"),
            displaced_points[1]
                .clamp(0.0, width_float - 1.0)
                .to_usize()
                .expect("failed to convert f32 to usize"),
        ];
        let (row, col) = (displaced_point_indices[0], displaced_point_indices[1]);
        let score = scores[(row, col, target_keypoint_id)];
        let image_coord = &displaced_point_indices
            * usize::from(self.output_stride)
            * offsets.slice(s![row, col, target_keypoint_id, ..]);
        Ok((
            score,
            image_coord.mapv(|v| v.to_f32().expect("failed to convert usize to f32")),
        ))
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
    ) -> Result<(Array1<f32>, Array2<f32>), Error> {
        let (.., num_parts) = scores.dim();

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
            let source_keypoint_id = source_keypoint_id.idx()?;
            let target_keypoint_id = target_keypoint_id.idx()?;
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
                )?;
                instance_keypoint_scores[target_keypoint_id] = score;
                instance_keypoint_coords
                    .slice_mut(s![target_keypoint_id, ..])
                    .assign(&coords);
            }
            Ok(())
        };

        pose::constants::POSE_CHAIN
            .iter()
            .rev()
            .enumerate()
            .try_for_each(|(edge_i, (target_keypoint_id, source_keypoint_id))| {
                apply(
                    edge_i,
                    source_keypoint_id,
                    target_keypoint_id,
                    displacements_bwd,
                )
            })?;

        pose::constants::POSE_CHAIN
            .iter()
            .enumerate()
            .try_for_each(move |(edge_i, (source_keypoint_id, target_keypoint_id))| {
                apply(
                    edge_i,
                    source_keypoint_id,
                    target_keypoint_id,
                    displacements_fwd,
                )
            })?;

        Ok((instance_keypoint_scores, instance_keypoint_coords))
    }

    fn within_nms_radius_fast(
        &self,
        pose_coords: ArrayView2<f32>,
        point: ArrayView1<f32>,
    ) -> Result<bool, Error> {
        let squared_nms_radius = self.nms_radius.pow(2).to_f32().ok_or(Error::ConvertToF32)?;
        Ok(pose_coords.dim().0 != 0
            && (&pose_coords - &point)
                .fold_axis(Axis(1), 0.0, |&total, &value| total + value.powi(2))
                .into_iter()
                .any(|v| v <= squared_nms_radius))
    }

    fn get_instance_score_fast(
        &self,
        // k x m x n
        exist_pose_coords: ArrayView3<f32>,
        keypoint_scores: ArrayView1<f32>,
        // m x n
        keypoint_coords: ArrayView2<f32>,
    ) -> Result<f32, Error> {
        let squared_nms_radius = self.nms_radius.pow(2).to_f32().ok_or(Error::ConvertToF32)?;
        let denominator = keypoint_scores.len().to_f32().ok_or(Error::ConvertToF32)?;
        let (first_dim, ..) = exist_pose_coords.dim();

        let sum = if first_dim != 0 {
            keypoint_scores
                .iter()
                .zip(
                    (&exist_pose_coords - &keypoint_coords)
                        .fold_axis(Axis(2), 0.0, |&total, &value| total + value.powi(2))
                        .fold_axis(Axis(0), true, |&previous, &value| {
                            previous && value > squared_nms_radius
                        }),
                )
                .filter_map(|(&v, m)| if m { Some(v) } else { None })
                .sum()
        } else {
            keypoint_scores.sum()
        };
        Ok(sum / denominator)
    }

    fn score_is_max_in_local_window(
        score: f32,
        y: usize,
        x: usize,
        local_maximum_radius: usize,
        scores: ArrayView2<f32>,
    ) -> bool {
        let (height, width) = scores.dim();
        let y_start = y.saturating_sub(local_maximum_radius);
        let y_end = height.min(y + local_maximum_radius + 1);
        let x_start = x.saturating_sub(local_maximum_radius);
        let x_end = width.min(x + local_maximum_radius + 1);
        scores
            .slice(s![y_start..y_end, x_start..x_end])
            .fold(true, |previous, &value| previous && value <= score)
    }

    fn build_part_with_score<'a>(
        &self,
        local_maximum_radius: usize,
        scores: &'a ArrayView3<'a, f32>,
    ) -> impl Iterator<Item = Part> + 'a {
        let score_threshold = self.score_threshold;
        scores
            .indexed_iter()
            .filter_map(move |((y, x, keypoint_id), &score)| {
                if score >= score_threshold
                    && Self::score_is_max_in_local_window(
                        score,
                        y,
                        x,
                        local_maximum_radius,
                        scores.index_axis(Axis(2), keypoint_id),
                    )
                {
                    Some(Part {
                        score,
                        keypoint_id,
                        x,
                        y,
                    })
                } else {
                    None
                }
            })
            .take(self.max_pose_detections)
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

        let mut scored_parts = self
            .build_part_with_score(pose::constants::LOCAL_MAXIMUM_RADIUS, &scores.view())
            .collect::<Vec<_>>();
        scored_parts
            .sort_by_key(|part| ordered_float::NotNan::new(part.score).expect("value is NaN"));

        let (width, height, _) = scores.dim();

        let new_shape = [width, height, 2, offsets.len() / (width * height * 2)];

        const TRANSPOSE_AXES: [usize; 4] = [0, 1, 3, 2];

        let new_offsets = offsets
            .into_shape(new_shape)
            .map_err(|e| Error::ReshapeOffsets(e, offsets.dim(), new_shape))?
            .permuted_axes(TRANSPOSE_AXES)
            .map(|&v| v.to_usize().unwrap());
        let new_displacments_fwd = displacements_fwd
            .permuted_axes(TRANSPOSE_AXES)
            .map(|&v| v.to_usize().unwrap());
        let new_displacments_bwd = displacements_bwd
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
            .map(|v| v.to_f32().expect("failed to convert usize to f32"));
            if !self.within_nms_radius_fast(
                pose_keypoint_coords.slice(s![..pose_count, keypoint_id, ..]),
                root_image_coord.view(),
            )? {
                let (keypoint_scores, keypoint_coords) = self.decode_pose(
                    score,
                    keypoint_id,
                    root_image_coord,
                    scores.view(),
                    new_offsets.view(),
                    new_displacments_fwd.view(),
                    new_displacments_bwd.view(),
                )?;
                pose_scores[pose_count] = self.get_instance_score_fast(
                    pose_keypoint_coords.slice(s![..pose_count, .., ..]),
                    keypoint_scores.view(),
                    keypoint_coords.view(),
                )?;
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

fn dequantize(v: u8, scale: f32, zero_point: f32) -> f32 {
    f32::from(v) * scale + zero_point
}

const HEATMAP_SCALE: f32 = 0.1157;
const HEATMAP_ZERO_POINT: f32 = 204.0;

const SHORT_OFFSETS_SCALE: f32 = 1.199291;
const SHORT_OFFSETS_ZERO_POINT: f32 = 134.0;

const MID_OFFSETS_SCALE: f32 = 2.820271;
const MID_OFFSETS_ZERO_POINT: f32 = 147.0;

impl crate::decode::Decoder for Decoder {
    fn expected_output_tensors(&self) -> usize {
        3
    }

    fn get_decoded_arrays<'a, 'b: 'a>(
        &'a self,
        interp: &'b mut tflite::Interpreter,
        (frame_width, frame_height): (usize, usize),
    ) -> Result<Box<[pose::Pose]>, Error> {
        let output_stride = usize::from(self.output_stride);
        let height = 1 + frame_height / output_stride;
        let width = 1 + frame_width / output_stride;

        let heatmaps = interp.get_output_tensor(0)?;
        let heatmaps = heatmaps
            .as_ndarray(
                unsafe { heatmaps.as_u8() },
                (width, height, pose::NUM_KEYPOINTS),
            )?
            .mapv(|v| 1.0 / (1.0 + (-dequantize(v, HEATMAP_SCALE, HEATMAP_ZERO_POINT)).exp()));

        let offsets = interp.get_output_tensor(1)?;
        let offsets = offsets
            .as_ndarray(
                unsafe { offsets.as_u8() },
                (width, height, 2 * pose::NUM_KEYPOINTS),
            )?
            .mapv(|v| dequantize(v, SHORT_OFFSETS_SCALE, SHORT_OFFSETS_ZERO_POINT));

        let raw_dsp = interp.get_output_tensor(2)?;
        let raw_dsp = raw_dsp
            .as_ndarray(
                unsafe { raw_dsp.as_u8() },
                (1, width, height, 4 * usize::from(self.output_stride)),
            )?
            .into_shape((width, height, 4, usize::from(self.output_stride)))
            .map_err(Error::ReshapeRawDsp)?
            .mapv(|v| dequantize(v, MID_OFFSETS_SCALE, MID_OFFSETS_ZERO_POINT));

        let fwd = raw_dsp.slice(s![.., .., ..2, ..]);
        let bwd = raw_dsp.slice(s![.., .., 2.., ..]);

        let (pose_scores, keypoint_scores, keypoints) =
            self.decode_multiple_poses(heatmaps, offsets.view(), fwd, bwd)?;

        crate::decode::reconstruct_from_arrays(
            keypoints.into(),
            keypoint_scores.into(),
            pose_scores.into(),
        )
    }
}
