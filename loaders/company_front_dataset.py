import os.path as osp

import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import Custom3DDataset


@DATASETS.register_module()
class CompanyFrontDataset(Custom3DDataset):
    """Front-view company dataset without nuScenes database dependencies."""

    CLASSES = (
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')

    def __init__(self, camera_key='CAM_FRONT', radar_key='RADAR_FRONT',
                 num_sweeps=7, **kwargs):
        self.camera_key = camera_key
        self.radar_key = radar_key
        self.num_sweeps = num_sweeps
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
