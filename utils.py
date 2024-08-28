import cv2
import numpy as np

def jigsaw_puzzle_solved_image(solved, nrows, ncols):
    # Create an empty list to store the rows of images
    rows = []

    # Loop through the array and resize and concatenate the images in each row
    for i in range(nrows):
        # Create an empty list to store the images in the current row
        row = []
        for j in range(ncols):
            row.append(solved[i][j].image)
        # Concatenate the images horizontally
        hori = np.concatenate(row, axis=1)
        # Append the concatenated row to the rows list
        rows.append(hori)

    # Concatenate the rows vertically
    vert = np.concatenate(rows, axis=0)
    return vert

def square_puzzle_solved_image(solved, nrows, ncols):
    # Create an empty list to store the rows of images
    rows = []

    # Loop through the array and resize and concatenate the images in each row
    for i in range(nrows):
        # Create an empty list to store the images in the current row
        row = []
        for j in range(ncols):
            img = cv2.resize(solved[i][j], (500, 500))
            row.append(img)
        # Concatenate the images horizontally
        hori = np.concatenate(row, axis=1)
        # Append the concatenated row to the rows list
        rows.append(hori)

    # Concatenate the rows vertically
    vert = np.concatenate(rows, axis=0)
    return vert

def show_images(imgs):
        for img in imgs:
            cv2.imshow("img", img)
            cv2.waitKey()

def show_image(img, factor=0.1):
    img = cv2.resize(img, None, fx=factor, fy=factor)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_image(path):
    img = cv2.imread(path)
    return img

def blur(img, filter_size=5):
    return cv2.GaussianBlur(img, (filter_size, filter_size), 0)


def filter(img):
    # currently only greenscreen filter
    RED, GREEN, BLUE = (2, 1, 0)

    reds = img[:, :, RED]
    greens = img[:, :, GREEN]
    blues = img[:, :, BLUE]

    mask = (greens < 35) | (reds > greens) | (blues > greens)
    return mask