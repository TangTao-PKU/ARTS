
import os
import cv2
import time
import json
import torch
import subprocess
import numpy as np
import os.path as osp
from pytube import YouTube
from collections import OrderedDict

from utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from utils._img_utils import get_single_image_crop_demo


def preprocess_video(video, joints2d, bboxes, frames, scale=1.0, crop_size=224):

    if joints2d is not None:
        bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
        bboxes[:,2:] = 150. / bboxes[:,2:]
        bboxes = np.stack([bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,2]]).T

        video = video[time_pt1:time_pt2]
        joints2d = joints2d[time_pt1:time_pt2]
        frames = frames[time_pt1:time_pt2]

    shape = video.shape

    temp_video = np.zeros((shape[0], crop_size, crop_size, shape[-1]))
    norm_video = torch.zeros(shape[0], shape[-1], crop_size, crop_size)

    for idx in range(video.shape[0]):

        img = video[idx]
        bbox = bboxes[idx]

        j2d = joints2d[idx] if joints2d is not None else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=scale,
            crop_size=crop_size)

        if joints2d is not None:
            joints2d[idx] = kp_2d

        temp_video[idx] = raw_img
        norm_video[idx] = norm_img

    temp_video = temp_video.astype(np.uint8)

    return temp_video, norm_video, bboxes, joints2d, frames


def download_youtube_clip(url, download_folder):
    return YouTube(url).streams.first().download(output_path=download_folder)


def trim_videos(filename, start_time, end_time, output_filename):
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    subprocess.call(command)


def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-r', '30000/1001',
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.jpg']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.jpg')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)


def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-framerate', '30000/1001', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.jpg', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):

    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def prepare_rendering_results(_results, nframes):
    frame_results = [{} for _ in range(nframes)]
    for person_id, person_data in _results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            frame_results[frame_id][person_id] = {
                'verts': person_data['mesh'][idx],
                'cam': person_data['pred_cam'][idx],
                'bbox': person_data['bboxes'][idx],
            }

    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['bbox'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results