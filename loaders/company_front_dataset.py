import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.datasets import DATASETS
from mmdet.core.evaluation.mean_ap import average_precision
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import Custom3DDataset


@DATASETS.register_module()
class CompanyFrontDataset(Custom3DDataset):
    """Front-view company dataset without nuScenes database dependencies."""

    CLASSES = (
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')

    def __init__(self, camera_key='CAM_FRONT', radar_key='RADAR_FRONT',
                 num_sweeps=7,
                 point_cloud_range=(0, -15, -3, 100, 15, 3), **kwargs):
        self.camera_key = camera_key
        self.radar_key = radar_key
        self.num_sweeps = num_sweeps
        self.point_cloud_range = np.asarray(
            point_cloud_range, dtype=np.float32)
        super().__init__(**kwargs)

    def load_annotations(self, ann_file):
        payload = mmcv.load(ann_file, file_format='pkl')
        self.metadata = payload.get('metadata', {})
        infos = payload['infos'] if isinstance(payload, dict) else payload
        return sorted(infos, key=lambda item: item['timestamp'])

    def _resolve_path(self, path):
        return path if osp.isabs(path) else osp.join(self.data_root, path)

    @staticmethod
    def _lidar2img(cam_info):
        if 'lidar2img' in cam_info:
            return np.asarray(cam_info['lidar2img'], dtype=np.float32)
        sensor2lidar_r = np.asarray(
            cam_info['sensor2lidar_rotation'], dtype=np.float32)
        sensor2lidar_t = np.asarray(
            cam_info['sensor2lidar_translation'], dtype=np.float32)
        lidar2cam_r = np.linalg.inv(sensor2lidar_r)
        lidar2cam_t = sensor2lidar_t @ lidar2cam_r.T
        lidar2cam = np.eye(4, dtype=np.float32)
        lidar2cam[:3, :3] = lidar2cam_r.T
        lidar2cam[3, :3] = -lidar2cam_t
        intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:3, :3] = intrinsic
        return viewpad @ lidar2cam.T

    def _camera_entry(self, cam_info):
        return dict(
            data_path=self._resolve_path(cam_info['data_path']),
            timestamp=int(cam_info['timestamp']),
            lidar2img=self._lidar2img(cam_info),
            cam_intrinsic=np.asarray(
                cam_info['cam_intrinsic'], dtype=np.float32))

    def _radar_entry(self, radar_info, fallback_timestamp):
        transform = radar_info.get(
            'radar2ego', radar_info.get('radar2lidar', np.eye(4)))
        return dict(
            data_path=self._resolve_path(radar_info['data_path']),
            timestamp=int(radar_info.get('timestamp', fallback_timestamp)),
            radar_in_ego=bool(radar_info.get('radar_in_ego', True)),
            radar2ego=np.asarray(transform, dtype=np.float32))

    def get_data_info(self, index):
        info = self.data_infos[index]
        cam = self._camera_entry(info['cams'][self.camera_key])
        radar = self._radar_entry(
            info['rads'][self.radar_key], info['timestamp'])

        camera_sweeps = []
        radar_sweeps = []
        for sweep in info.get('sweeps', [])[:self.num_sweeps]:
            if self.camera_key in sweep:
                camera_sweeps.append(
                    self._camera_entry(sweep[self.camera_key]))
            if self.radar_key in sweep:
                radar_sweeps.append(self._radar_entry(
                    sweep[self.radar_key], info['timestamp']))

        input_dict = dict(
            sample_idx=info['token'],
            timestamp=info['timestamp'] / 1e6,
            pts_filename=self._resolve_path(info['lidar_path']),
            lidar_in_ego=bool(info.get('lidar_in_ego', True)),
            lidar2ego=np.asarray(info.get('lidar2ego', np.eye(4)), dtype=np.float32),
            front_camera=cam,
            camera_sweeps=camera_sweeps,
            front_radar=radar,
            radar_sweeps=radar_sweeps,
            img_filename=[cam['data_path']],
            img_timestamp=[cam['timestamp'] / 1e6],
            lidar2img=[cam['lidar2img']],
            intrinsics=[cam['cam_intrinsic']])
        if not self.test_mode:
            input_dict['ann_info'] = self.get_ann_info(index)
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_boxes = np.asarray(
            info.get('gt_boxes', np.zeros((0, 7))), dtype=np.float32)
        gt_velocity = np.asarray(
            info.get('gt_velocity', np.zeros((len(gt_boxes), 2))),
            dtype=np.float32).reshape(-1, 2)
        gt_names = np.asarray(info.get('gt_names', []))
        valid = np.asarray(
            info.get('valid_flag', np.ones(len(gt_boxes), dtype=bool)),
            dtype=bool)
        gt_boxes = gt_boxes[valid]
        gt_velocity = gt_velocity[valid]
        gt_names = gt_names[valid]
        class_to_id = {name: idx for idx, name in enumerate(self.CLASSES)}
        gt_labels = np.array(
            [class_to_id.get(name, -1) for name in gt_names], dtype=np.int64)
        known = gt_labels >= 0
        gt_boxes = gt_boxes[known]
        gt_velocity = gt_velocity[known]
        gt_names = gt_names[known]
        gt_labels = gt_labels[known]
        if gt_boxes.shape[1] == 7:
            gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=1)
        boxes = LiDARInstance3DBoxes(
            gt_boxes, box_dim=gt_boxes.shape[-1] if gt_boxes.ndim == 2 else 7,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        return dict(
            gt_bboxes_3d=boxes,
            gt_labels_3d=gt_labels,
            gt_names=gt_names,
            gt_names_3d=gt_names)

    @staticmethod
    def _result_fields(result):
        result = result.get('pts_bbox', result)
        return (
            result['boxes_3d'],
            result['scores_3d'].detach().cpu().numpy(),
            result['labels_3d'].detach().cpu().numpy())

    @staticmethod
    def _bev_iou(boxes1, boxes2):
        if len(boxes1) == 0 or len(boxes2) == 0:
            return torch.zeros(
                (len(boxes1), len(boxes2)), dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor1 = boxes1.tensor[:, [0, 1, 3, 4, 6]].to(device=device)
        tensor2 = boxes2.tensor[:, [0, 1, 3, 4, 6]].to(device=device)
        return box_iou_rotated(tensor1, tensor2).detach().cpu()

    @staticmethod
    def _iou_3d(boxes1, boxes2):
        if len(boxes1) == 0 or len(boxes2) == 0:
            return torch.zeros(
                (len(boxes1), len(boxes2)), dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        boxes1 = boxes1.to(device)
        boxes2 = boxes2.to(device)
        return LiDARInstance3DBoxes.overlaps(boxes1, boxes2).detach().cpu()

    def _filtered_gt(self, index):
        ann = self.get_ann_info(index)
        boxes = ann['gt_bboxes_3d']
        labels = ann['gt_labels_3d']
        if len(boxes) == 0:
            return boxes, labels
        centers = boxes.gravity_center
        roi = self.point_cloud_range
        mask = (
            (centers[:, 0] >= roi[0]) & (centers[:, 0] <= roi[3]) &
            (centers[:, 1] >= roi[1]) & (centers[:, 1] <= roi[4]) &
            (centers[:, 2] >= roi[2]) & (centers[:, 2] <= roi[5]))
        return boxes[mask], labels[mask.cpu().numpy()]

    def _evaluate_class(self, predictions, ground_truth, class_id,
                        iou_fn, iou_threshold, score_threshold):
        gt_by_frame = {}
        num_gt = 0
        for frame_id, (boxes, labels) in enumerate(ground_truth):
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            gt_by_frame[frame_id] = class_boxes
            num_gt += len(class_boxes)

        detections = []
        overlaps_by_frame = {}
        for frame_id, (boxes, scores, labels) in enumerate(predictions):
            indices = np.flatnonzero(labels == class_id)
            class_boxes = boxes[indices.tolist()]
            overlaps_by_frame[frame_id] = iou_fn(
                class_boxes, gt_by_frame[frame_id]).numpy()
            for local_id, index in enumerate(indices):
                detections.append((float(scores[index]), frame_id, local_id))
        detections.sort(key=lambda item: item[0], reverse=True)

        matched = {
            frame_id: np.zeros(len(boxes), dtype=bool)
            for frame_id, boxes in gt_by_frame.items()}
        true_positive = np.zeros(len(detections), dtype=np.float32)
        false_positive = np.zeros(len(detections), dtype=np.float32)

        for det_id, (_, frame_id, local_id) in enumerate(detections):
            gt_boxes = gt_by_frame[frame_id]
            if len(gt_boxes) == 0:
                false_positive[det_id] = 1.0
                continue
            overlaps = overlaps_by_frame[frame_id][local_id].copy()
            overlaps[matched[frame_id]] = -1.0
            gt_id = int(overlaps.argmax())
            if overlaps[gt_id] >= iou_threshold:
                true_positive[det_id] = 1.0
                matched[frame_id][gt_id] = True
            else:
                false_positive[det_id] = 1.0

        tp_cumulative = np.cumsum(true_positive)
        fp_cumulative = np.cumsum(false_positive)
        recalls = tp_cumulative / max(num_gt, np.finfo(np.float32).eps)
        precisions = tp_cumulative / np.maximum(
            tp_cumulative + fp_cumulative, np.finfo(np.float32).eps)
        ap = float(average_precision(recalls, precisions, mode='area')) \
            if num_gt > 0 and len(detections) > 0 else 0.0

        threshold_count = sum(score >= score_threshold for score, _, _ in detections)
        threshold_tp = float(true_positive[:threshold_count].sum())
        threshold_fp = float(false_positive[:threshold_count].sum())
        precision = threshold_tp / max(threshold_tp + threshold_fp, 1.0)
        recall = threshold_tp / max(float(num_gt), 1.0)
        return dict(
            ap=ap, precision=precision, recall=recall,
            num_gt=num_gt, num_predictions=len(detections),
            threshold_tp=threshold_tp, threshold_fp=threshold_fp)

    def evaluate(self, results, metric=None, logger=None,
                 bev_iou_threshold=0.5, iou_3d_threshold=0.5,
                 score_threshold=0.1, **kwargs):
        """Evaluate front-view detections with class-wise BEV and 3D AP."""
        if len(results) != len(self):
            raise ValueError(
                f'Expected {len(self)} predictions, received {len(results)}')

        predictions = [self._result_fields(result) for result in results]
        ground_truth = [self._filtered_gt(index) for index in range(len(self))]
        metrics = {}
        bev_aps = []
        iou_3d_aps = []
        total_tp = 0.0
        total_fp = 0.0
        total_gt = 0

        for class_id, class_name in enumerate(self.CLASSES):
            bev = self._evaluate_class(
                predictions, ground_truth, class_id, self._bev_iou,
                bev_iou_threshold, score_threshold)
            iou_3d = self._evaluate_class(
                predictions, ground_truth, class_id, self._iou_3d,
                iou_3d_threshold, score_threshold)
            if bev['num_gt'] == 0:
                continue
            bev_aps.append(bev['ap'])
            iou_3d_aps.append(iou_3d['ap'])
            total_tp += bev['threshold_tp']
            total_fp += bev['threshold_fp']
            total_gt += bev['num_gt']
            prefix = f'company/{class_name}'
            metrics[f'{prefix}_BEV_AP@{bev_iou_threshold:g}'] = bev['ap']
            metrics[f'{prefix}_3D_AP@{iou_3d_threshold:g}'] = iou_3d['ap']
            metrics[f'{prefix}_precision@{score_threshold:g}'] = bev['precision']
            metrics[f'{prefix}_recall@{score_threshold:g}'] = bev['recall']
            metrics[f'{prefix}_num_gt'] = bev['num_gt']

        metrics[f'company/BEV_mAP@{bev_iou_threshold:g}'] = \
            float(np.mean(bev_aps)) if bev_aps else 0.0
        metrics[f'company/3D_mAP@{iou_3d_threshold:g}'] = \
            float(np.mean(iou_3d_aps)) if iou_3d_aps else 0.0
        metrics[f'company/overall_precision@{score_threshold:g}'] = \
            total_tp / max(total_tp + total_fp, 1.0)
        metrics[f'company/overall_recall@{score_threshold:g}'] = \
            total_tp / max(float(total_gt), 1.0)
        metrics['company/total_gt'] = total_gt
        return metrics
