import cv2
import imutils
import numpy as np


def is_contour_valid(contour, img):
    x, y, w, h = cv2.boundingRect(contour)
    bbox = img.copy()
    cv2.rectangle(bbox, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # test if contour touches sides of image
    return x == 0 or y == 0 or x + w == 256 or y + h == 256


def filter_blobs(mask):
    blobs = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blobs = imutils.grab_contours(blobs)
    mask = np.ones(mask.shape[:2], dtype=int) * 255

    for c in blobs:
        if not is_contour_valid(c, mask):
            cv2.drawContours(mask, [c], -1, 0, -1)

    return np.array((((np.ones(mask.shape[:2], dtype=int) * 255) - mask) / 255)).astype(np.uint8)
