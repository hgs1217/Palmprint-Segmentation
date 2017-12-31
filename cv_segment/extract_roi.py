# @Create Date: 2017/12/21

import cv2
import numpy as np
import cv_segment.utils as utils

PATH = "tmp/"


def output_distance_transform():
    img = cv2.imread("pics/test{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
    x, bi_src = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    trans = cv2.distanceTransform(bi_src, cv2.DIST_L2, 5)
    cv2.imwrite(PATH + "trans.jpg", trans)
    return img, trans


def should_rotate_degree(img):
    # img = cv2.imread("tmp/src_trans.jpg", 0)
    _, binary = cv2.threshold(img, int(img.max() * 0.80), 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(binary, 1, 2)
    [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
    return int(np.arctan(vy / vx) / np.pi * 180)
    # print(vx, vy, x, y)
    # lefty = int((-x * vy / vx) +
    # righty = int(((img.shape[1] - x) * vy / vx) + y)
    # img = cv2.line(img, (img.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)


def new_distance_transform(img, img_binary):
    # img = cv2.imread("tmp/rotated_ori.jpg", cv2.IMREAD_GRAYSCALE)
    # x, img_binary = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    trans = cv2.distanceTransform(img_binary, cv2.DIST_L2, 5)
    cv2.imwrite(PATH + "rotated_trans.jpg", trans)
    return trans


def find_center(img):
    max_pixel = img.max()
    _, interest = cv2.threshold(img, max_pixel - 2, 255, cv2.THRESH_BINARY)
    yl, xl = np.nonzero(interest)
    cx = int(np.average(xl))
    cy = int(np.average(yl))
    # print(cx, cy)
    return (cx, cy)

    # max_pixel = img.max()
    # x = y = 0
    # max_l = []
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         # if img[i][j] == 255: points.append((i,j))
    #         if img[i][j] >= max_pixel - 1:
    #             max_l.append((i, j))
    #             x += j
    #             y += i
    # cx = x // len(max_l)
    # cy = y // len(max_l)
    # print(cx, cy)


def is_right_hand(img):
    h, w = img.shape[:2]
    if np.sum(img[50, 0:w]) > np.sum(img[h - 50, 0:w]):
        print("left hand detected.")
        return False
    else:
        return True


def _extract_roi(src_ori, i):
    # if not is_right_hand(src_ori):
    #     src_ori = cv2.flip(src_ori, 0)

    _, src_binary = cv2.threshold(src_ori, 60, 255, cv2.THRESH_BINARY)
    src_trans = cv2.distanceTransform(src_binary, cv2.DIST_L2, 5)
    src_trans = src_trans.astype(np.uint8)
    # cv2.imwrite(PATH + "src_trans.jpg", src_trans)

    rotate_angle = should_rotate_degree(src_trans)
    rotated_ori = utils.rotate_bound(src_ori, -rotate_angle)  ##???
    rotated_ori = rotated_ori.astype(np.uint8)
    # cv2.imwrite("tmp/rotated_ori.jpg", rotated_ori)

    _, rotated_binary = cv2.threshold(rotated_ori, 60, 255, cv2.THRESH_BINARY)
    rotated_binary = rotated_binary.astype(np.uint8)

    rotated_trans = cv2.distanceTransform(rotated_binary, cv2.DIST_L2, 5)
    cv2.imwrite(PATH + "rotated_trans{}.jpg".format(i), rotated_trans)
    (cx, cy) = find_center(rotated_trans)

    _, rotated_trans_binary = cv2.threshold(rotated_trans, 55, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("tmp/rotated_trans_binary.jpg", rotated_trans_binary)

    center_line = rotated_trans_binary[cy, 0: rotated_ori.shape[1]]
    # print(center_line)
    white_xs = np.nonzero(center_line)[0]
    most_left = int(white_xs[0])
    most_right = min(int(cx + (cx - most_left) * 0.6), int(white_xs[-1]))

    roi_center_x = (most_left + cx) // 2
    roi_center_column = rotated_trans_binary[0:rotated_ori.shape[0], roi_center_x]
    overall_center_column = rotated_binary[0:rotated_ori.shape[0], int(most_left * 0.7 + 0.3 * cx)]

    white_ys = np.nonzero(roi_center_column)[0]
    most_top = np.nonzero(overall_center_column)[0][0]
    most_bottom = np.nonzero(overall_center_column)[0][-1]

    cut_range = (most_left, most_bottom, most_right, most_top)
    print("ROI:LBRT-", cut_range)

    show_roi_rectangle = rotated_ori.copy()
    cv2.rectangle(show_roi_rectangle, (most_left, most_top), (most_right, most_bottom), (0, 0, 0))

    pure_roi = np.zeros([most_bottom - most_top + 1, most_right - most_left + 1])
    pure_roi = rotated_ori[most_top: most_bottom + 1, most_left: most_right + 1]

    roi_for_cnn = utils.resize(pure_roi, 128, 128)

    return roi_for_cnn, pure_roi, show_roi_rectangle, rotate_angle, cut_range


def mapping(orix, roi_res, rotate_degree, cut_range, need_flip):
    if need_flip:
        ori = cv2.flip(orix, 0)
    else:
        ori = orix.copy()
    width = cut_range[2] - cut_range[0]
    height = cut_range[1] - cut_range[3]
    src_h, src_w = roi_res.shape[:2]
    roi_normal_size = utils.resize_for_roi(roi_res, src_w, src_h, width, height)
    rotated = utils.rotate_bound(ori, -rotate_degree)
    if np.ndim(rotated) == 2 or np.ndim(rotated) == 3 and rotated.shape[2] == 1:
        rotate_add = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)
    else:
        rotate_add = rotated

    (x1, y2, x2, y1) = cut_range
    aux = np.zeros((int(rotated.shape[0]), int(rotated.shape[1])))
    # print(roi_normal_size.shape)
    aux[y1:y2, x1:x2] = roi_normal_size
    rotate_add[aux > 0] = [0, 0, 255]

    rotate_back = utils.rotate_bound(rotate_add, rotate_degree)

    top = cut_vertical = (rotate_back.shape[0] - ori.shape[0]) // 2
    left = cut_horizon = (rotate_back.shape[1] - ori.shape[1]) // 2
    bottom = rotate_back.shape[0] - cut_vertical
    right = rotate_back.shape[1] - cut_horizon

    rotate_back = rotate_back[top:bottom, left:right]
    if need_flip:
        return cv2.flip(rotate_back, 0)
    else:
        return rotate_back


def get_roi(image, img_seg, i):
    need_flip = False
    if not is_right_hand(image):
        src_ori = cv2.flip(image, 0)
        need_flip = True
    else:
        src_ori = image.copy()
    roi_for_cnn, roi, show_roi, angle, cut_range = _extract_roi(src_ori, i)

    if need_flip:
        rote = cv2.flip(img_seg, 0)
        rote = utils.rotate_bound(rote, -angle)
    else:
        rote = utils.rotate_bound(img_seg, -angle)
    (x1, y2, x2, y1) = cut_range
    cut_preseg = rote[y1:y2, x1:x2]
    cut_preseg = utils.resize(cut_preseg, 128, 128)
    return roi_for_cnn, roi, show_roi, angle, cut_range, need_flip, cut_preseg


if __name__ == '__main__':
    for i in range(13, 14):
        img = cv2.imread("pics/r{}.jpg".format(i - 12), cv2.IMREAD_GRAYSCALE)
        img_seg = img

        # GET ROI
        roi_for_cnn, roi, show_roi, angle, cut_range, need_flip, cut_seg = get_roi(img, img_seg, 0)
        cv2.imwrite("roi/roi{}.jpg".format(i), roi_for_cnn)
        cv2.imwrite("roi/seg{}.jpg".format(i), cut_seg)
        cv2.imwrite("roi/show_roi{}.jpg".format(i), show_roi)

        # SIMPLE SEGMENTATION
        # _, roi_res = cv2.threshold(roi_for_cnn, 100, 255, cv2.THRESH_BINARY_INV)
        roi_res = cv2.imread("pics/r3.jpg", cv2.IMREAD_GRAYSCALE)
        final = mapping(img, roi_res, angle, cut_range, need_flip)
        cv2.imwrite("pics/final{}.jpg".format(i), final)
