import numpy as np
import cv2
from classes.puzzle_class import Puzzle
from classes.square_puzzle_class import SquarePuzzle
import utils

#img = utils.read_image("images/jigsaw_puzzle.jpg")
#hint = utils.read_image("images/wallpaper.jpg")
#puzzle = Puzzle(img, hint=hint)
#solved = puzzle.hint_solver()

#img = utils.read_image("images/jigsaw_puzzle.jpg")
#puzzle = Puzzle(img)
#solved = puzzle.solver()

img = utils.read_image("images/grid_puzzle200.jpg")
height, width, channels = img.shape
hint = utils.read_image("images/wallpaper.jpg")
rows = 10
cols = 20
puzzle = SquarePuzzle(img, rows, cols, True, hint)
solved = puzzle.hint_solver()

# to show solved jigsaw puzzle, use utils.jigsaw_puzzle_solved_image instead
solved_image = utils.square_puzzle_solved_image(solved, puzzle.rows, puzzle.cols)
cv2.imwrite("solved.jpg", solved_image) 

