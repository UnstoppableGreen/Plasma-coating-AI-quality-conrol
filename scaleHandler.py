import math

import cv2
import easyocr
import numpy as np
import tensorflow as tf
from skimage.feature import hog

scaleMap = [100, 200, 300, 400, 500]


def calculate_pixels_per_square_micron(scale_length, scale_value, shape0, shape1):
    height_in_micrometers = shape0 / scale_length * scale_value

    width_in_micrometers = shape1 / scale_length * scale_value

    pixels_per_micron_height = shape0 / height_in_micrometers

    pixels_per_micron_width = shape1 / width_in_micrometers

    pixels_per_square_micron = pixels_per_micron_height * pixels_per_micron_width

    return pixels_per_square_micron


def find_scale_length_and_value(image):
    x_mid = image.shape[1] / 2
    y_mid = image.shape[0] / 2

    left_bot_quarter = image[(int(y_mid) + int(y_mid * 0.3)):image.shape[0], 0:(int(x_mid * 0.7))].copy()
    if find_line_length(left_bot_quarter):
        return (find_line_length(left_bot_quarter), find_value(left_bot_quarter))

    right_bot_quarter = image[int(y_mid * 1.2):image.shape[0], int(x_mid * 1.2):image.shape[1]].copy()
    if (find_line_length(right_bot_quarter)):
        return (find_line_length(right_bot_quarter), find_value(right_bot_quarter))


def find_line_length(quarter):
    gray = cv2.cvtColor(quarter, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        rho=1,  # Distance resolution in pixels
        theta=np.pi / 180,  # Angle resolution in radians
        threshold=50,  # Min number of votes for valid line
        minLineLength=100,  # Min allowed length of line
        maxLineGap=3  # Max allowed gap between line for joining them
    )
    dists_list = []

    if lines is not None:
        for points in lines:
            x1, y1, x2, y2 = points[0]
            dists_list.append(math.hypot(x2 - x1, y2 - y1))
    else:
        return False

    # return ((max(dists_list)+statistics.fmean(dists_list))/2)
    return max(dists_list)


def find_value(bot_quarter):
    model = tf.keras.models.load_model('./models/CNN_Model.h5')
    # clf = joblib.load("./models/digits_cls.pkl")

    ## convert to hsv
    hsv = cv2.cvtColor(bot_quarter, cv2.COLOR_BGR2HSV)
    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv2.inRange(hsv, (29, 128, 130), (
        71, 255, 255))  # yellow mask = cv2.inRange(hsv, (44, 50, 70), (100, 255,255)) (30, 140, 145), (96, 255,255)
    ## slice colours
    imask = mask > 0
    masked = np.zeros_like(bot_quarter, np.uint8)
    masked[imask] = bot_quarter[imask]
    ##split grayscale
    h, s, v = cv2.split(masked)
    ##noise filtering
    ret, thres = cv2.threshold(v, 10, 100, cv2.THRESH_BINARY_INV)
    ##reconstruction
    kernel = np.ones((2, 2), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
    masked = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    ##whitenning background
    _, masked = cv2.threshold(masked, 90, 255, cv2.THRESH_BINARY_INV)

    reader = easyocr.Reader(['en'], gpu=False)
    ocr_digits = list(map(int, reader.readtext(masked, detail=0, allowlist='0123456789')))
    digits0 = []
    for number in ocr_digits:
        for x in str(number):
            digits0.append(int(x))

    ctrs, hier = cv2.findContours(masked.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = list(filter(lambda cnt: cv2.contourArea(cnt) > 25, ctrs))

    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    number = ' '
    digits1 = []
    digits2 = []

    for rect in rects:
        length = int(rect[3] * 1.4)
        pt1 = int(rect[1] + rect[3] // 2 - length // 2 - 2)
        pt2 = int(rect[0] + rect[2] // 2 - length // 2 - 2)
        roi = masked[pt1:pt1 + length + 2, pt2:pt2 + length + 2]

        try:
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            # Recognize digit using CNN
            t = roi.reshape(1, 28, 28, 1)
            predict_x = model.predict(t)
            classes_x = np.argmax(predict_x, axis=1)
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
            # nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
            digits1.append(classes_x[0])
            # digits2.append(nbr[0])
        except:
            print('problem happened')
    print(digits0)
    print(digits1)
    print(digits2)
    threeDigits = ''
    for digit in digits0:
        threeDigits += str(digit)
        if len(threeDigits) == 3:
            if int(threeDigits) in scaleMap:
                return int(threeDigits)
            threeDigits = ''

    if 1 in list(set(digits1).intersection(digits0)):
        return 100
    if 3 in list(set(digits1).intersection(digits0)):
        return 300
    if not digits1 and not digits0:
        return -1
