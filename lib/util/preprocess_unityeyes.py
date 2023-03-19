import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from util.gazemap import from_gaze2d

def preprocess_unityeyes_image(img, json_data):
    img = np.array(img, dtype=np.uint8)
    ow = 60
    oh = 36
    ih, iw = img.shape[:2]
    ih_2, iw_2 = ih/2.0, iw/2.0

    heatmap_w = int(ow)
    heatmap_h = int(oh)

    def process_coords(coords_list):
        coords = [eval(l) for l in coords_list]
        return np.array([(x, 36-y, z) for (x, y, z) in coords])

    def preprocess_eyeball_center(coords_list):
        coords = [eval(l) for l in coords_list]
        return np.array([(x, 36-y) for (y, x) in coords])


    interior_landmarks = process_coords(json_data['interior_margin_2d'])
    caruncle_landmarks = process_coords(json_data['caruncle_2d'])
    iris_landmarks = process_coords(json_data['iris_2d'])
    eyeball_center = preprocess_eyeball_center(json_data['eyeball_center'])

    gaze = np.array(eval(json_data['eye_details']['look_vec']))[:3].reshape((1, 3))
    gaze = gaze.astype(np.float32)
    iris_center = np.mean(iris_landmarks[:, :2], axis=0)
    landmarks = np.concatenate([interior_landmarks[:, :2],  # 8
                                iris_landmarks[::2, :2],  # 8
                                iris_center.reshape((1, 2)),
                                eyeball_center,
                                ])  # 18 in total

    temp = np.zeros((34, 2), dtype=np.float32)
    temp[:, 0] = landmarks[:, 1]
    temp[:, 1] = landmarks[:, 0]
    landmarks = temp
    heatmaps = get_heatmaps(w=heatmap_w, h=heatmap_h, landmarks=landmarks)
    heatmaps = np.asarray(heatmaps)
    assert heatmaps.shape == (34, heatmap_h, heatmap_w)
    # print('gaze.shape: ',gaze.shape)
    # gaze_2d = [gaze[0][1]*-1, gaze[0][0]*-1]
    # gazemap = from_gaze2d(gaze_2d, output_size=(ih, iw), scale=1.0,).astype(np.float32)

    # return heatmaps, landmarks, gaze, gazemap
    return heatmaps, landmarks, gaze

def gaussian_2d(w, h, cx, cy, sigma=2.0):
    """Generate heatmap with single 2D gaussian."""
    xs, ys = np.meshgrid(
        np.linspace(0, w - 1, w, dtype=np.float32),
        np.linspace(0, h - 1, h, dtype=np.float32)
    )

    assert xs.shape == (h, w)
    alpha = -0.5 / (sigma ** 2)
    heatmap = np.exp(alpha * ((xs - cx) ** 2 + (ys - cy) ** 2))
    return heatmap


def get_heatmaps(w, h, landmarks):
    heatmaps = []
    for (y, x) in landmarks:
        heatmaps.append(gaussian_2d(w, h, cx=x, cy=y, sigma=2.0))
    return np.array(heatmaps)
