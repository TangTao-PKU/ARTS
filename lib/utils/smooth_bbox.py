
import numpy as np
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d


def get_smooth_bbox_params(kps, vis_thresh=2, kernel_size=11, sigma=3):

    bbox_params, start, end = get_all_bbox_params(kps, vis_thresh)
    smoothed = smooth_bbox_params(bbox_params, kernel_size, sigma)
    smoothed = np.vstack((np.zeros((start, 3)), smoothed))
    return smoothed, start, end


def kp_to_bbox_param(kp, vis_thresh):

    if kp is None:
        return
    vis = kp[:, 2] > vis_thresh
    if not np.any(vis):
        return
    min_pt = np.min(kp[vis, :2], axis=0)
    max_pt = np.max(kp[vis, :2], axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height < 0.5:
        return
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height
    return np.append(center, scale)


def get_all_bbox_params(kps, vis_thresh=2):
    # keeps track of how many indices in a row with no prediction
    num_to_interpolate = 0
    start_index = -1
    bbox_params = np.empty(shape=(0, 3), dtype=np.float32)

    for i, kp in enumerate(kps):
        bbox_param = kp_to_bbox_param(kp, vis_thresh=vis_thresh)
        if bbox_param is None:
            num_to_interpolate += 1
            continue

        if start_index == -1:
            # Found the first index with a prediction!
            start_index = i
            num_to_interpolate = 0

        if num_to_interpolate > 0:
            # Linearly interpolate each param.
            previous = bbox_params[-1]
            # This will be 3x(n+2)
            interpolated = np.array(
                [np.linspace(prev, curr, num_to_interpolate + 2)
                 for prev, curr in zip(previous, bbox_param)])
            bbox_params = np.vstack((bbox_params, interpolated.T[1:-1]))
            num_to_interpolate = 0
        bbox_params = np.vstack((bbox_params, bbox_param))

    return bbox_params, start_index, i - num_to_interpolate + 1


def smooth_bbox_params(bbox_params, kernel_size=11, sigma=8):

    smoothed = np.array([signal.medfilt(param, kernel_size)
                         for param in bbox_params.T]).T
    return np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T