import numpy as np
import cv2
from classes.piece_class import Piece
from classes.piece_class import PieceType, EdgeType
import utils
from classes.compare_images import CompareImage
from skimage import color

class Puzzle:
    def __init__(self, img, hint = None):
        self.image = img
        self.pieces = self.get_pieces()
        self.corner_pieces = [piece for piece in self.pieces if piece.piece_type == PieceType.CORNER]
        self.edge_pieces = [piece for piece in self.pieces if piece.piece_type == PieceType.EDGE]
        self.middle_pieces = [piece for piece in self.pieces if piece.piece_type == PieceType.MIDDLE]
        self.cols, self.rows = self.puzzle_dimensions()
        self.images = None
        self.hint = hint
    
    def show_statics(self):
        
        def draw_points_on_image(image, points, color=(0, 255, 0), thickness=2):
            """
            Draws points on an image.

            Parameters:
            image (numpy.ndarray): The image on which to draw the points.
            points (list of tuples): A list of (x, y) tuples representing the points.
            color (tuple): The color of the points (B, G, R).
            thickness (int): The thickness of the points.
            """
            for point in points:
                cv2.circle(image, point, radius=1, color=color, thickness=thickness)
            return image
        drawn_imgs = []
        for piece in self.pieces:
            img = piece.image
            for corner_point in piece.corners:
                    cv2.circle(img, corner_point, radius=3, color=(0, 0, 0), thickness=1)
            drawn_imgs.append(img)
        for img in drawn_imgs:
            utils.show_image(img, 0.5)

            
    
    def puzzle_dimensions(self):
        total_edges = len(self.edge_pieces) + 2 * len(self.corner_pieces)
        P = total_edges/4
        total_pieces = len(self.edge_pieces) + len(self.middle_pieces) + len(self.corner_pieces)
        Q = np.sqrt(P**2 - total_pieces)
        cols = P + Q
        rows = P - Q
        return int(cols), int(rows)
    
    def split_grids(self, image):

            image = np.array(image) # convert the image to a numpy array
            images = np.array_split(image, self.rows, axis=0) # split the array along the rows
            images = [np.array_split(row, self.cols, axis=1) for row in images] # split each row along the columns

            return images


    def show_images(self):
        for piece in self.pieces:
            cv2.imshow("Piece Image", piece.image)
            cv2.waitKey()

    def get_pieces(self):
        blurred = utils.blur(self.image)
        mask = utils.filter(blurred)
        binary_mask = (mask > 0).astype(np.uint8)
        thresh = self.threshold_image(binary_mask)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return self.crop_pieces(closing)

    def calculate_scale(self, target):
        width, height = len(self.image[0]), len(self.image)
        return target / width if height > width else target / height

    def threshold_image(self, binary_mask):
        _, thresh = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def remove_false_detections(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        unique, counts = unique[1:], counts[1:]  # Remove background
        q75, q25 = np.percentile(counts, [75, 25])
        iqr = q75 - q25
        low, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        return [label for label, count in zip(unique, counts) if low < count < hi], \
               [count for count in counts if low < count < hi]

    def get_bounding_boxes(self, clean_labels, stats, margin, scale):
        boxes = []
        for label in clean_labels:
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                          stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            box = self.calculate_box(x, y, w, h, margin, scale)
            boxes.append(np.multiply(box, 1 / scale))
        return boxes

    def calculate_box(self, x, y, w, h, margin, scale):
        if h > w:
            return ((x - (h - w) / 2 - margin, y - margin),
                    (x + h + margin - (h - w) / 2, y + h + margin))
        else:
            return ((x - margin, y - margin - (w - h) / 2),
                    (x + w + margin, y + w + margin - (w - h) / 2))

    def crop_pieces(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        margin = 100
        # Find the maximum width and height of all the contours
        max_width = 0
        max_height = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            max_width = max(max_width, w + 2 * margin)
            max_height = max(max_height, h + 2 * margin)

        pieces = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Make sure the margin doesn't go beyond the image boundaries
            x_margin = max(x - margin, 0)
            y_margin = max(y - margin, 0)
            w_margin = min(w + 2 * margin, self.image.shape[1] - x_margin)
            h_margin = min(h + 2 * margin, self.image.shape[0] - y_margin)
            # Crop with added margin
            cropped_piece_with_margin = self.image[y_margin:y_margin+h_margin, x_margin:x_margin+w_margin]
            # Pad the cropped image with black pixels to match the maximum width and height
            padded_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            padded_image[:h_margin, :w_margin] = cropped_piece_with_margin
            # Change black pixels to green
            padded_image[np.where((padded_image==[0,0,0]).all(axis=2))] = [72,145,29]
            piece = Piece(padded_image)
            pieces.append(piece)
        return pieces

    def ciede2000_distance(self, color1, color2):
        """Calculate the CIEDE2000 color difference between two colors."""
        return color.deltaE_ciede2000(color1, color2)

    def match_metric(self, piece, prev, edge_color1, edge_color2):

        lab1 = piece.image_colors
        lab2 = prev.image_colors
        # Calculate the delta e for each pixel pair using the CIEDE2000 formula
        delta_e = self.ciede2000_distance(lab1, lab2)

        # Find the average delta e for the whole images
        avg_delta_e = np.mean(delta_e)
        min_length = min(len(edge_color1), len(edge_color2))
        distances = []        
        for i in range(min_length):
            color1 = edge_color1[i]
            color2 = edge_color2[i]
            distance = np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))
            distances.append(distance)
        avg_distance = np.mean(distances)
        max_dist = max(distances)
        w1 = 0.9
        w2 = 0.1
        # Define the maximum possible delta e value
        # For CIE L*a*b*, it is about 100
        delta_e_max = 100

        score = w1 * (avg_delta_e / delta_e_max) + w2 * (avg_distance / max_dist)
        return score
 

    
    def solver(self):
        print("Started Solving the Puzzle: -->")
        # Initialize an empty array to store the solved puzzle
        solved = np.zeros((self.rows, self.cols), dtype=object)

        for piece in self.corner_pieces:
            # Find the top-left corner piece and place it in the solved array
            if piece.edge_types.count(EdgeType.FLAT) == 2 and piece.edge_types[0] == EdgeType.FLAT and piece.edge_types[3] == EdgeType.FLAT:
                solved[0][0] = piece
            # Find the bottom-left corner piece and place it in the solved array
            if piece.edge_types.count(EdgeType.FLAT) == 2 and piece.edge_types[0] == EdgeType.FLAT and piece.edge_types[1] == EdgeType.FLAT:
                solved[0][self.cols - 1] = piece
            # Find the top-right corner piece and place it in the solved array
            if piece.edge_types.count(EdgeType.FLAT) == 2 and piece.edge_types[2] == EdgeType.FLAT and piece.edge_types[3] == EdgeType.FLAT:
                solved[self.rows - 1][0] = piece
            # Find the bottom-right corner piece and place it in the solved array
            if piece.edge_types.count(EdgeType.FLAT) == 2 and piece.edge_types[1] == EdgeType.FLAT and piece.edge_types[2] == EdgeType.FLAT:
                solved[self.rows - 1][self.cols - 1] = piece
        
        # Find the rest of the pieces in the top row and place them in the solved array
        for col in range(1, self.cols):
            if col == self.cols - 1:
                break
            # Find the piece that matches the right edge of the previous piece
            prev_piece = solved[0][col-1]
            # Get the average color of the right edge of the prev_piece
            prev_color = prev_piece.edges_colors[1]
            # Initialize the minimum distance and the best piece
            min_dist = np.inf
            best_piece = None
            for piece in self.edge_pieces:
                if (piece != prev_piece) and (piece not in solved) and piece.edge_types.count(EdgeType.FLAT) == 1 and piece.edge_types[0] == EdgeType.FLAT and piece.match_edge(prev_piece, 3, 1):
                    # Get the average color of the left edge of the piece
                    piece_color = piece.edges_colors[3]
                    # Calculate the Euclidean distance between the two colors
                    dist = self.match_metric(piece, prev_piece, piece_color, prev_color) #np.sqrt(np.sum((np.array(prev_color) - np.array(piece_color)) ** 2))
                    # Update the minimum distance and the best piece if a better match is found
                    if dist < min_dist:
                        min_dist = dist
                        best_piece = piece
            # Assign the best piece to the solved array
            solved[0][col] = best_piece
        
        
        # Find the rest of the pieces in the Bottom row and place them in the solved array
        for col in range(1, self.cols):
            if col == self.cols - 1:
                break
            # Find the piece that matches the right edge of the previous piece
            prev_piece = solved[self.rows - 1][col-1]
            # Get the average color of the right edge of the prev_piece
            prev_color = prev_piece.edges_colors[1]
            # Initialize the minimum distance and the best piece
            min_dist = np.inf
            best_piece = None
            for piece in self.edge_pieces:
                if (piece != prev_piece) and (piece not in solved) and piece.edge_types.count(EdgeType.FLAT) == 1 and piece.edge_types[2] == EdgeType.FLAT and piece.match_edge(prev_piece, 3, 1):
                    # Get the average color of the left edge of the piece
                    piece_color = piece.edges_colors[3]
                    # Calculate the Euclidean distance between the two colors
                    dist = self.match_metric(piece, prev_piece, piece_color, prev_color)
                    # Update the minimum distance and the best piece if necessary
                    if dist < min_dist:
                        min_dist = dist
                        best_piece = piece
            # Assign the best piece to the solved array
            solved[self.rows - 1][col] = best_piece


        # Find the rest of the pieces in the left column and place them in the solved array
        for row in range(1, self.rows):
            if row == self.rows - 1:
                break
            # Find the piece that matches the bottom edge of the previous piece
            prev_piece = solved[row-1][self.cols - 1]
            # Get the average color of the right edge of the prev_piece
            prev_color = prev_piece.edges_colors[2]
            # Initialize the minimum distance and the best piece
            min_dist = np.inf
            best_piece = None
            for piece in self.edge_pieces:
                if (piece != prev_piece) and (piece not in solved) and piece.edge_types.count(EdgeType.FLAT) == 1 and piece.edge_types[1] == EdgeType.FLAT and piece.match_edge(prev_piece, 0, 2):
                    # Get the average color of the left edge of the piece
                    piece_color = piece.edges_colors[0]
                    # Calculate the Euclidean distance between the two colors
                    dist = self.match_metric(piece, prev_piece, piece_color, prev_color)
                    # Update the minimum distance and the best piece if necessary
                    if dist < min_dist:
                        min_dist = dist
                        best_piece = piece
            # Assign the best piece to the solved array
            solved[row][self.cols-1] = best_piece

        
        # Find the rest of the pieces in the right column and place them in the solved array
        for row in range(1, self.rows):
            if row == self.rows - 1:
                break
            # Find the piece that matches the bottom edge of the previous piece
            prev_piece = solved[row-1][0]
            # Get the average color of the right edge of the prev_piece
            prev_color = prev_piece.edges_colors[2]
            # Initialize the minimum distance and the best piece
            min_dist = np.inf
            best_piece = None
            for piece in self.edge_pieces:
                if (piece != prev_piece) and (piece not in solved) and piece.edge_types.count(EdgeType.FLAT) == 1 and piece.edge_types[3] == EdgeType.FLAT and piece.match_edge(prev_piece, 0, 2):
                    # Get the average color of the left edge of the piece
                    piece_color = piece.edges_colors[0]
                    # Calculate the Euclidean distance between the two colors
                    dist = self.match_metric(piece, prev_piece, piece_color, prev_color)
                    # Update the minimum distance and the best piece if necessary
                    if dist < min_dist:
                        min_dist = dist
                        best_piece = piece
            # Assign the best piece to the solved array
            solved[row][0] = best_piece

        ## Find the rest of the pieces in the middle and place them in the solved array
        for row in range(1, self.rows):
            if row == self.rows - 1:
                break
            for col in range(1, self.cols):
                if col == self.cols - 1:
                    break
                # Find the piece that matches the right edge of the previous piece and the bottom edge of the above piece
                prev_piece = solved[row][col-1]
                above_piece = solved[row-1][col]
                # Get the average color of the right edge of the prev_piece and the bottom edge of the above_piece
                prev_color = prev_piece.edges_colors[1]
                above_color = above_piece.edges_colors[2]
                # Initialize the minimum distance and the best piece
                min_dist = np.inf
                best_piece = None
                for piece in self.middle_pieces:
                    if (piece != prev_piece) and (piece != above_piece) and (piece not in solved) and (piece.match_edge(prev_piece, 3, 1)) and piece.match_edge(above_piece, 0, 2):
                        # Get the average color of the left edge and the top edge of the piece
                        piece_color1 = piece.edges_colors[3]
                        piece_color2 = piece.edges_colors[0]
                        # Calculate the Euclidean distance between the two colors
                        dist1 = dist = self.match_metric(piece, prev_piece, piece_color1, prev_color)
                        dist2 = dist = self.match_metric(piece, above_piece, piece_color2, above_color)
                        # Sum the two distances
                        dist = dist1 + dist2
                        # Update the minimum distance and the best piece if necessary
                        if dist < min_dist:
                            min_dist = dist
                            best_piece = piece
                # Assign the best piece to the solved array
                solved[row][col] = best_piece

        # Return the solved array
        return solved
    
    def hint_solver(self):
        self.images = self.split_grids(self.hint)
        print("Started Solving the Puzzle with Hint: -->")
        # Initialize an empty array to store the solved puzzle
        solved = np.zeros((self.rows, self.cols), dtype=object)

        for piece in self.corner_pieces:
            # Find the top-left corner piece and place it in the solved array
            if piece.edge_types.count(EdgeType.FLAT) == 2 and piece.edge_types[0] == EdgeType.FLAT and piece.edge_types[3] == EdgeType.FLAT:
                solved[0][0] = piece
            # Find the top-right corner piece and place it in the solved array
            if piece.edge_types.count(EdgeType.FLAT) == 2 and piece.edge_types[0] == EdgeType.FLAT and piece.edge_types[1] == EdgeType.FLAT:
                solved[0][self.cols - 1] = piece
            # Find the bottom-left corner piece and place it in the solved array
            if piece.edge_types.count(EdgeType.FLAT) == 2 and piece.edge_types[2] == EdgeType.FLAT and piece.edge_types[3] == EdgeType.FLAT:
                solved[self.rows - 1][0] = piece
            if piece.edge_types.count(EdgeType.FLAT) == 2 and piece.edge_types[1] == EdgeType.FLAT and piece.edge_types[2] == EdgeType.FLAT:
            # Find the bottom-right corner piece and place it in the solved array
                solved[self.rows - 1][self.cols - 1] = piece
        
        # Find the pieces in the top row and place them in the solved array
        for col in range(1, self.cols):
            if col == self.cols - 1:
                break
            # Find the piece that matches the right edge of the previous piece
            prev_piece = solved[0][col-1]
            # Get the average color of the right edge of the prev_piece
            hint_piece = self.images[0][col]
            # Initialize the minimum distance and the best piece
            min_res = np.inf
            for piece in self.edge_pieces:
                if (piece != prev_piece) and (piece not in solved) and piece.edge_types.count(EdgeType.FLAT) == 1 and piece.edge_types[0] == EdgeType.FLAT and piece.match_edge(prev_piece, 3, 1):
                    res = CompareImage(hint_piece, piece.image).compare_images_features()
                    if res == False:
                        res = CompareImage(piece.image, hint_piece).compare_image()
                        if res < min_res:
                            min_res = res
                            solved[0][col] = piece
                            #utils.show_image(piece.image)
                    else:
                        solved[0][col] = piece
                        break
        # Bottom edges
        for col in range(1, self.cols):
            if col == self.cols - 1:
                break
            # Find the piece that matches the right edge of the previous piece
            prev_piece = solved[self.rows - 1][col-1]
            hint_piece = self.images[self.rows - 1][col]
            # Initialize the minimum distance and the best piece
            min_res = np.inf
            for piece in self.edge_pieces:
                if (piece != prev_piece) and (piece not in solved) and piece.edge_types.count(EdgeType.FLAT) == 1 and piece.edge_types[2] == EdgeType.FLAT and piece.match_edge(prev_piece, 3, 1):
                    res = CompareImage(hint_piece, piece.image).compare_images_features()
                    if res == False:
                        res = CompareImage(piece.image, hint_piece).compare_image()
                        if res < min_res:
                            min_res = res
                            solved[self.rows-1][col] = piece
                            #utils.show_image(piece.image)
                    else:
                        solved[self.rows - 1][col] = piece
                        break

        for row in range(1, self.rows):
            if row == self.rows - 1:
                break
            # Find the piece that matches the bottom edge of the previous piece
            prev_piece = solved[row-1][self.cols - 1]
            hint_piece = self.images[row][self.cols - 1]
            # Get the average color of the right edge of the prev_piece
            # Initialize the minimum distance and the best piece
            min_res = np.inf
            for piece in self.edge_pieces:
                if (piece != prev_piece) and (piece not in solved) and piece.edge_types.count(EdgeType.FLAT) == 1 and piece.edge_types[1] == EdgeType.FLAT and piece.match_edge(prev_piece, 0, 2):
                    res = CompareImage(hint_piece, piece.image).compare_images_features()
                    if res == False:
                        res = CompareImage(piece.image, hint_piece).compare_image()
                        if res < min_res:
                            min_res = res
                            solved[row][self.cols-1] = piece
                    else:
                        solved[row][self.cols -1] = piece
                        break
        for row in range(1, self.rows):
            if row == self.rows - 1:
                break
            # Find the piece that matches the bottom edge of the previous piece
            prev_piece = solved[row-1][0]
            hint_piece = self.images[row][0]
            # Get the average color of the right edge of the prev_piece
            # Initialize the minimum distance and the best piece
            min_res = np.inf
            for piece in self.edge_pieces:
                if (piece != prev_piece) and (piece not in solved) and piece.edge_types.count(EdgeType.FLAT) == 1 and piece.edge_types[3] == EdgeType.FLAT and piece.match_edge(prev_piece, 0, 2):
                    res = CompareImage(hint_piece, piece.image).compare_images_features()
                    if res == False:
                        res = CompareImage(piece.image, hint_piece).compare_image()
                        if res < min_res:
                            min_res = res
                            solved[row][0] = piece
                    else:
                        solved[row][0] = piece
                        break

        for row in range(1, self.rows):
            if row == self.rows - 1:
                break
            for col in range(1, self.cols):
                if col == self.cols - 1:
                    break
                # Find the piece that matches the right edge of the previous piece and the bottom edge of the above piece
                prev_piece = solved[row][col-1]
                above_piece = solved[row-1][col]
                hint_piece = self.images[row][col]

                # Get the average color of the right edge of the prev_piece and the bottom edge of the above_piece
                # Initialize the minimum distance and the best piece
                min_res = np.inf
                for piece in self.middle_pieces:
                    if (piece != prev_piece) and (piece != above_piece) and (piece not in solved) and (piece.match_edge(prev_piece, 3, 1)) and piece.match_edge(above_piece, 0, 2):
                        res = CompareImage(hint_piece, piece.image).compare_images_features()
                        if res == False:
                            res = CompareImage(piece.image, hint_piece).compare_image()
                            if res < min_res:
                                min_res = res
                                solved[row][col] = piece
                        else:
                            solved[row][col] = piece
                            break
            # Assign the best piece to the solved array
        return solved
