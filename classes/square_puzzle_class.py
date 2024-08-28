import numpy as np
from classes.compare_images import CompareImage

class SquarePuzzle():
    def __init__(self, image, rows, cols, is_hint=False, hint=None):
        self.image = image
        self.is_hint = is_hint
        self.hint = hint
        self.rows = rows
        self.cols = cols
        self.pieces = self.split_grids(self.image)
        if is_hint:
            self.images = self.split_grids(hint)

    def split_grids(self, image):

            image = np.array(image) # convert the image to a numpy array
            images = np.array_split(image, self.rows, axis=0) # split the array along the rows
            images = [np.array_split(row, self.cols, axis=1) for row in images] # split each row along the columns
            return images
    

    def hint_solver(self):
        # initialize an empty dictionary to store the matches
        solved = [[0 for j in range(self.cols)] for i in range(self.rows)]
        # loop over the pieces
        for i in range(len(self.images)):
            for j in range(len(self.images[0])):
                # loop over the images
                min_res = np.inf
                for m in range(len(self.pieces)):
                    for n in range(len(self.pieces[0])):
                        image = self.images[i][j]
                        piece = self.pieces[m][n]
                        compare = CompareImage(image, piece).compare_image()
                        if compare < min_res:
                            solved[i][j] = piece
                            min_res = compare
                        continue

        # return the solved array
        return solved


    
