import numpy as np
import cv2
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import utils
def show_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_corners(mask, harris_blocksize):
    gray = np.float32(mask)
    dst = cv2.cornerHarris(gray, harris_blocksize, 31, 0.04)
    dst = cv2.dilate(dst, None)
    dst = dst * gray
    data = dst.copy()
    data[data < 0.4 * data.max()] = 0
    datamax = filters.maximum_filter(data, 5)
    maxima = (dst == datamax)
    datamin = filters.minimum_filter(data, 5)
    diff = ((datamax - datamin) > .01)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))
    temp = yx[:, 0].copy()
    yx[:, 0] = yx[:, 1]
    yx[:, 1] = temp
    return np.round(yx).astype(np.int)


def get_angle(p1,p2,p3):
    """calculate angle between p2_p3 and p2_p3"""
    p2p1 = get_vector(p2,p1)
    p2p3 = get_vector(p2,p3)
    cosine_angle = np.dot(p2p1, p2p3) / (np.linalg.norm(p2p1) * np.linalg.norm(p2p3))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def get_distance(pt1,pt2):
    return ((pt2[0]-pt1[0]) ** 2 + (pt2[1]-pt1[1]) ** 2) ** (1/2)

def get_90deg_corners(pt, corners, rule1, rule2, margin = 5):
    """calculate all corners that are 90 degrees from pt, where the corners c1, c2 must comply with rule1, rule2 """
    res = []
    used = []
    for c1 in corners:
        if not rule1(c1):
            continue
        for c2 in corners:
            if np.array_equal(c1,c2) or np.array_equal(c2,pt) or np.array_equal(c1,pt) or \
                    not rule2(c2) or \
                    tuple(c2) in used:
                continue
            if 90 - margin < get_angle(c1,pt,c2) < 90 + margin:
                res.append((c1,c2))
                used.append(tuple(c1))
    return res



def get_vector(p1,p2):
    if type(p1).__module__ != np.__name__:
        p1 = np.array(p1)
    if type(p2).__module__ != np.__name__:
        p2 = np.array(p2)
    return p2-p1

def get_tile_corners(mask, harris_blocksize=20, angle_margin=10, side_len_var_thres = 1000, angle_diff_ls_thres=100, rec_level=10):
    """
    returns points of tile corners
    :param mask: tile mask
    :param crop: croped image (for debugging)
    :param harris_blocksize: initial block size for harris algorithm (will be increased upto rec_level*2 if no corners are found)
    :param angle_margin: 90 +- angle_margin degrees are allowed
    :param side_len_var_thres: corners are accepted if the side lenght variance is under this number
    :param angle_diff_ls_thres: corners are accepted if the least squares angle diff to 90 is under this
    :param rec_level: how many time the harris blocksize should be increased by 2
    :return: array of 4 points
    """
    corners = get_corners(mask, harris_blocksize)
    tile_center = ndimage.center_of_mass(mask)
    tile_center = tuple(np.round(tile_center).astype(np.int))
    ang_opt = np.array([90, 90, 90, 90])
    ang_diff_ls = None
    side_var = None
    tile_corners = []

    for c1 in corners:
        if c1[0] <= tile_center[0] and c1[1] <= tile_center[1]:
            # identify candidates for top left corner

            candidates1 = get_90deg_corners(c1, corners,
                                            lambda c: c[0] <= tile_center[0] and c[1] >= tile_center[1],
                                            lambda c: c[0] >= tile_center[0] and c[1] <= tile_center[1], angle_margin)

            for c2, c4 in candidates1:
                for c3 in corners:
                    if c3[0] >= tile_center[0] and c3[1] >= tile_center[1]:
                        # identify candidates for bottom right corner
                        candidates2 = get_90deg_corners(c3, corners,
                                                        lambda c: True,
                                                        # c[0] <= tile_center[0] and c[1] >= tile_center[1],
                                                        lambda c: True,
                                                        angle_margin)  # c[0] >= tile_center[0] and c[1] <= tile_center[1])
                        for t2, t4 in candidates2:
                            if (((np.array_equal(c2, t2) and np.array_equal(c4, t4)) or
                                 (np.array_equal(c2, t4) and np.array_equal(c4, t2)))) and 90 - angle_margin < get_angle(c2,c3,c4) < 90 + angle_margin and 90 - angle_margin < get_angle(c3, c4, c1) < 90 + angle_margin:
                                new = [c1, c2, c3, c4]
                                ang_new = np.array([get_angle(new[i], new[(i + 1) % 4], new[(i + 2) % 4]) for i in range(4)])
                                ang_diff_ls_new = np.sum(np.square(ang_opt - ang_new))
                                side_var_new = np.var([get_distance(new[(i + 1) % 4], new[i]) for i in range(4)])
                                if len(tile_corners) > 0:
                                    # check if better match
                                    # by comparing least squares residual of angles to insure rectangle
                                    # and of variance of distance to center to insure a centered rectangle
                                    # and of variance of the side lengths to insure square

                                    dist_var_new = np.var([get_distance(x, tile_center) for x in new])
                                    dist_var_curr = np.var([get_distance(tile_corners[i], tile_center) for i in range(4)])
                                    if ang_diff_ls_new > ang_diff_ls or dist_var_new > dist_var_curr or side_var_new > side_var:
                                        continue
                                elif ang_diff_ls_new > angle_diff_ls_thres or side_var_new > side_len_var_thres:
                                    continue
                                ang_diff_ls = ang_diff_ls_new
                                side_var = side_var_new
                                tile_corners = [c1, c2, c3, c4]
    if len(tile_corners) == 0:
        if rec_level <= 0:
            raise RuntimeError(f"Could not detect tile corners! (with max harris blocksize {harris_blocksize})")
        return get_tile_corners(mask, harris_blocksize+2, angle_margin, side_len_var_thres, angle_diff_ls_thres, rec_level-1)
    return improve_tile_corners(tile_corners, tile_center, mask)


def improve_tile_corners(tile_corners, tile_center, mask):
    """Improves corners by finding the furthest TRUE point in the mask from the corner in direction center->corner"""
    imp_tile_corners = []
    for c in tile_corners:
        v = get_vector(tile_center, c)
        v = v/v[0] #scale vector to y=1
        if c[0]>tile_center[0]:
            f = lambda x: (c + (v * x)).astype(int)
        else:
            f = lambda x: (c - (v * x)).astype(int)
        d = 0
        pt = f(d+1)
        while mask[pt[1],pt[0]]:
            d += 1
            pt = f(d+1)
        imp_tile_corners.append(f(d))
    return imp_tile_corners

import numpy as np
import cv2

def order_points(pts):
    ### modified code from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

