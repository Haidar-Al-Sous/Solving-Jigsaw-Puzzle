import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from classes.puzzle_class import Puzzle
from classes.square_puzzle_class import SquarePuzzle
import utils

root = tk.Tk()
root.attributes()

puzzle_path = None
hint_path = None
square_path = None
square_hint_path = None
solved = None
puzzle = None
square_puzzle = None

# Define functions for button actions
def choose_image(is_hint=False, is_square=False, is_square_hint = False):
    global puzzle_path
    global hint_path

    global square_path
    global square_hint_path

    global puzzle
    global square_puzzle

    filename = filedialog.askopenfilename(title="Choose Puzzle Image")
    if filename:
        # Display the chosen puzzle image
        puzzle_image = Image.open(filename)
        puzzle_image = resize_image(puzzle_image)
        puzzle_photo = ImageTk.PhotoImage(puzzle_image)
        puzzle_image_label.configure(image=puzzle_photo)
        puzzle_image_label.image = puzzle_photo  # Keep a reference to prevent garbage collection

        if square_path != None and square_hint_path == None:
            square_hint_path = filename

        if is_square == False:
            choose_hint_button.config(state="normal")
            solve_puzzle_button.config(state="normal")
            choose_square_hint_button.config(state="disabled")
            if is_hint:
                hint_path = filename
                return
            else:
                puzzle_path = filename
                return
        else:
            square_path = filename
            choose_square_hint_button.config(state="normal")
            choose_puzzle_button.config(state="disabled")
            return

def show_statics():
    global puzzle
    puzzle.show_statics()

def solve_puzzle():
    global solved
    global puzzle
    global puzzle_path
    global square_path
    global hint_path
    global square_hint_path

    if square_path == None:
        if hint_path != None:     
            puzzle_img = utils.read_image(puzzle_path)
            hint_img = utils.read_image(hint_path)
            puzzle = Puzzle(puzzle_img, hint_img)   
            solved = puzzle.hint_solver()
            show_statics_button.config(state="normal")
            utils.show_image(utils.jigsaw_puzzle_solved_image(solved, puzzle.rows, puzzle.cols))
        else:
            puzzle_img = utils.read_image(puzzle_path)
            puzzle = Puzzle(puzzle_img)
            solved = puzzle.solver()
            show_statics_button.config(state="normal")
            utils.show_image(utils.jigsaw_puzzle_solved_image(solved, puzzle.rows, puzzle.cols))
    else:
        puzzle_img = utils.read_image(square_path)
        hints_img = utils.read_image(square_hint_path)
        rows = 9
        cols = 9
        puzzle = SquarePuzzle(image=puzzle_img,
                              rows=rows,
                              cols=cols,
                              is_hint=True,
                              hint=hints_img)
        solved = puzzle.hint_solver()
        utils.show_image(utils.square_puzzle_solved_image(solved, puzzle.rows, puzzle.cols))
    


def resize_image(image, height_ratio=0.5):
    width, height = image.size
    aspect_ratio = width / height
    new_height = int(root.winfo_height() * height_ratio)
    new_width = int(new_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height))
    return resized_image    

# Updated background colors
root.configure(bg="#333333")  # Dark gray for the app
side_panel = tk.Frame(root, width=300, bg="#555555")  # Increase side panel width and use gray color

side_panel.pack(side=tk.LEFT, fill=tk.Y)

choose_puzzle_button = tk.Button(side_panel, text="Choose Puzzle", command=choose_image)
choose_puzzle_button.pack(pady=10, padx=80)

choose_square_button = tk.Button(side_panel, text="Choose Square Puzzle", command=lambda: choose_image(is_square=True))
choose_square_button.pack(pady=10, padx=80)

choose_hint_button = tk.Button(side_panel, text="Choose Hint", command=lambda: choose_image(True), state="disabled")
choose_hint_button.pack(pady=10)

choose_square_hint_button = tk.Button(side_panel, text="Choose Hint Square", command=lambda: choose_image(True), state="disabled")
choose_square_hint_button.pack(pady=10)

show_statics_button = tk.Button(side_panel, text="Show Statics", command=show_statics, state="disabled")
show_statics_button.pack(pady=10)

# Button to solve the puzzle (optional)
solve_puzzle_button = tk.Button(side_panel, text="Solve Puzzle", command=solve_puzzle, state="disabled")
solve_puzzle_button.pack(pady=10)

# Updated image panel placements with padding
puzzle_image_label = tk.Label(root, pady=50)
puzzle_image_label.pack()

solved_puzzle_image_label = tk.Label(root, pady=50)
solved_puzzle_image_label.pack()

root.mainloop()
