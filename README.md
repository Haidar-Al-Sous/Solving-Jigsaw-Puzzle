# Jigsaw Puzzle Solving using Computer Vision

## Introduction

This project aims to solve Jigsaw puzzles using computer vision techniques. It supports both traditional Jigsaw puzzles and square puzzles, with or without hints.

## Features

- **Jigsaw Puzzle Support:** Solves puzzles by identifying corner, edge, and middle pieces.
- **Square Puzzle Support:** Solves puzzles with square pieces, using a hint if provided.
- **Image Processing:** Utilizes techniques like Gaussian Blur, contour detection, and edge classification.
- **Flexible Input:** Handles both images with complete pieces and those with hints.

## Technical Overview

- **Languages & Tools:** Implemented using Python and OpenCV.
- **Algorithms:** Includes Harris corner detection, Canny edge detection, and template matching.
- **Efficiency:** Designed to mimic human problem-solving approaches for improved efficiency.

## How to Use

1. **Input:** Provide a puzzle image with a green background (for Jigsaw) or a grid image and hint (for Square puzzles).
2. **Execution:** Run the program to process and solve the puzzle.
3. **Output:** The solved puzzle image is displayed.

## Limitations

- Pieces must not be rounded or overlapping.
- Background color must be green for Jigsaw puzzles.
- Tested on puzzles with regular shapes.

## Performance

| Type | Hint | Number of Pieces | Accuracy | Execution Time |
|------|------|------------------|----------|----------------|
| Jigsaw | False | 32 | ~60% | 46 seconds |
| Jigsaw | True | 32 | ~84% | 26 seconds |
| Grid | True | 200 | 100% | 9 seconds |
| Grid | True | 450 | 100% | 30 seconds |

This project was developed as part of a research initiative to explore computer vision applications in puzzle solving.

![image](https://github.com/user-attachments/assets/09d35bd8-61bf-4fed-8338-aedfcfed9a4c)

![image](https://github.com/user-attachments/assets/11679a39-3c1f-46ca-b617-4bd82896572f)


