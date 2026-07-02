import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type


def _load_float_points(path, load_dim):
    if path.endswith('.npy'):
        points = np.load(path)
    else:
        points = np.fromfile(path, dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, load_dim)


def _roi_mask(points, roi):
    x_min, y_min, z_min, x_max, y_max, z_max = roi
    return ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max))


@PIPELINES.register_module()
class LoadFrontCameraSweeps:
    """Load one front camera for the current frame and temporal sweeps."""

    def __init__(self, sweeps_num=7, color_type='color'):
        self.sweeps_num = sweeps_num
        self.color_type = color_type

    def __call__(self, results):
        entries = [results['front_camera']]
        sweeps = results.get('camera_sweeps', [])[:self.sweeps_num]
        if not sweeps:
            sweeps = [results['front_camera']] * self.sweeps_num
        elif len(sweeps) < self.sweeps_num:
            sweeps = sweeps + [sweeps[-1]] * (self.sweeps_num - len(sweeps))
        entries.extend(sweeps)

        results['img'] = [
            mmcv.imread(entry['data_path'], self.color_type) for entry in entries]
        results['filename'] = [entry['data_path'] for entry in entries]
        results['img_timestamp'] = [entry['timestamp'] / 1e6 for entry in entries]
        results['lidar2img'] = [
            np.asarray(entry['lidar2img'], dtype=np.float32) for entry in entries]
        results['intrinsics'] = []
        for entry in entries:
            intrinsic = np.eye(4, dtype=np.float32)
            intrinsic[:3, :3] = entry['cam_intrinsic']
            results['intrinsics'].append(intrinsic)
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadCompanyLidarPoints:
    def __init__(self, load_dim=5, use_dim=5,
                 roi=(0, -20, -3, 200, 20, 3)):
        self.load_dim = load_dim
        self.use_dim = list(range(use_dim)) if isinstance(use_dim, int) else use_dim
        self.roi = roi

    def __call__(self, results):
        points = _load_float_points(results['pts_filename'], self.load_dim)
        points = points[:, self.use_dim]
        if not results.get('lidar_in_ego', True):
            transform = np.asarray(results['lidar2ego'], dtype=np.float32)
            xyz1 = np.concatenate(
                [points[:, :3], np.ones((len(points), 1), dtype=np.float32)],
                axis=1)
            points[:, :3] = (xyz1 @ transform.T)[:, :3]
        points = points[_roi_mask(points, self.roi)]
        points_class = get_points_type('LIDAR')
        results['points'] = points_class(points, points_dim=points.shape[1])
        return results


@PIPELINES.register_module()
class LoadCompanyRadarSweeps:
    """Normalize company radar to ego, then apply the front ROI."""

    def __init__(self, sweeps_num=7, load_dim=7,
                 roi=(0, -20, -3, 200, 20, 3)):
        self.sweeps_num = sweeps_num
        self.load_dim = load_dim
        self.roi = roi

    def _load_entry(self, entry, reference_timestamp):
        points = _load_float_points(entry['data_path'], self.load_dim)
        if not entry.get('radar_in_ego', True):
            transform = np.asarray(entry['radar2ego'], dtype=np.float32)
            xyz1 = np.concatenate(
                [points[:, :3], np.ones((len(points), 1), dtype=np.float32)],
                axis=1)
            points[:, :3] = (xyz1 @ transform.T)[:, :3]
            if points.shape[1] >= 6:
                velocity = np.concatenate(
                    [points[:, 4:6], np.zeros((len(points), 1), dtype=np.float32)],
                    axis=1)
                points[:, 4:6] = (velocity @ transform[:3, :3].T)[:, :2]
        if points.shape[1] >= 7:
            points[:, 6] = (reference_timestamp - entry['timestamp']) / 1e6
        return points[_roi_mask(points, self.roi)]

    def __call__(self, results):
        current = results['front_radar']
        entries = [current]
        sweeps = results.get('radar_sweeps', [])[:self.sweeps_num]
        if not sweeps:
            sweeps = [current] * self.sweeps_num
        elif len(sweeps) < self.sweeps_num:
            sweeps = sweeps + [sweeps[-1]] * (self.sweeps_num - len(sweeps))
        entries.extend(sweeps)

        points_class = get_points_type('LIDAR')
        results['radar_points'] = []
        for entry in entries:
            points = self._load_entry(entry, current['timestamp'])
            results['radar_points'].append(
                points_class(points, points_dim=points.shape[1]))
        return results


@PIPELINES.register_module()
class FrontViewFilter:
    """Apply one front ROI consistently to LiDAR, radar, and GT boxes."""

    def __init__(self, roi=(0, -20, -3, 200, 20, 3)):
        self.roi = roi

    def __call__(self, results):
        if 'points' in results:
            mask = _roi_mask(results['points'].tensor, self.roi)
            results['points'] = results['points'][mask]
        if 'radar_points' in results:
            filtered = []
            for points in results['radar_points']:
                mask = _roi_mask(points.tensor, self.roi)
                filtered.append(points[mask])
            results['radar_points'] = filtered
        if 'gt_bboxes_3d' in results:
            centers = results['gt_bboxes_3d'].gravity_center
            mask = _roi_mask(centers, self.roi)
            results['gt_bboxes_3d'] = results['gt_bboxes_3d'][mask]
            mask_np = mask.cpu().numpy()
            for key in ('gt_labels_3d', 'gt_names_3d'):
                if key in results:
                    results[key] = results[key][mask_np]
        return results
