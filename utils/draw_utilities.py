import numpy as np
import cv2


def show_mask(mask_, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask_.shape[-2:]
    mask_image = mask_.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def draw_points(rec_image, points):
    for _point in points:
        rec_image = cv2.circle(rec_image, (_point[0], _point[1]), 10, (255, 255, 0), -1)

    return rec_image
