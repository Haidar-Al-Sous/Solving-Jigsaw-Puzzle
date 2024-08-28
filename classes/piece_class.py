from skimage import color as cr
import classes.corner_utils as corner_utils
import utils
import numpy as np
import cv2
import math
from enum import Enum, auto
from skimage import color

class PieceType(Enum):
    CORNER = auto()
    EDGE = auto()
    MIDDLE = auto()

class EdgeType(Enum):
    FLAT = auto()
    HEAD = auto()
    HOLE = auto()

def show_image(img1, dst2):
    img = img1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = dst2
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
class Piece:
    def __init__(self, image):
        self.image = image
        self.mask = utils.filter(utils.blur(self.image, 21))
        self.corners = np.array(corner_utils.get_tile_corners(self.mask))
        self.canny_edge = self.get_canny_edge()
        self.piece_type = None
        self.boundaries = []
        self.edges_colors = []
        self.edge_types = []
        self.image_colors = None
        self.extract_image_colors()
        self.draw_and_label_edges(0.0001)

    # This will help us find the closest point between each corner
    # and the canny edge, because sometimes, and due to some noises
    # in the preprocessing of the images, some corners tend to be
    # a little bit away from the canny edge, but with this function
    # we approximate each corner to the nearest point on the edge.
    def find_closest_points(self, points_of_interest, big_set_of_points):
        # Convert the lists to NumPy arrays for efficient calculations
        points_of_interest = np.asarray(points_of_interest)
        big_set_of_points = np.asarray(big_set_of_points)

        # Calculate distances from each point in the big set to each point of interest
        distances = np.linalg.norm(big_set_of_points[:, np.newaxis] - points_of_interest, axis=2)
        
        # Find the closest point for each point of interest
        closest_indices = np.argmin(distances, axis=0)

        # Assign the closest points to each point of interest
        closest_points = big_set_of_points[closest_indices]

        closest_points = [tuple(point) for point in closest_points]
        return closest_points
    
    # This method will extract the colors of the image in
    # CIE L*a*b* color space
    def extract_image_colors(self):
        img = self.image.copy()
        # Apply the mask to the image to get the foreground color
        fg = cv2.bitwise_and(img, img, mask=self.mask.astype(np.uint8))

        # Convert the foreground color to CIE L*a*b* color space
        lab = color.rgb2lab(fg)

        self.image_colors = lab
    
    # Sorting the corners clockwise (or at least sorting the corners)
    # will help us find each edge of the piece, because each edge is
    # the points between two adjecent edges.
    def sort_points_clockwise(self, points):
        # Calculate the centroid of the points
        centroid_x = sum(point[0] for point in points) / len(points)
        centroid_y = sum(point[1] for point in points) / len(points)

        # Define a function to calculate the angle between the centroid and a point
        def angle_from_centroid(point):
            return math.atan2(point[1] - centroid_y, point[0] - centroid_x)

        # Sort the points based on the angle from the centroid
        sorted_points = sorted(points, key=angle_from_centroid)

        return sorted_points
    
    # classifies an edge based on its curvature and shape,
    # categorizing it as either a "head," "hole," or "flat" edge.
    def classify_edge(self, points, threshold, NUM):
        # Convert points to numpy array for efficient calculations
        points = np.array(points)

        # Calculate the line equation from the first and last point
        x_coords, y_coords = points[:,0], points[:,1]
        line_coeffs = np.polyfit(x_coords, y_coords, 1)

        # Calculate the rotation angle to align points on the x-axis
        angle = np.arctan(NUM * line_coeffs[0])
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # Rotate all points
        aligned_points = np.dot(points - points[0], rotation_matrix)

        # Calculate the curvature of the edge
        curvature = np.polyfit(aligned_points[:,0], aligned_points[:,1], 2)

        # Calculate the signed area under the curve
        signed_area = 0
        for i in range(1, len(aligned_points)):
            signed_area += 0.5 * (aligned_points[i-1][0] + aligned_points[i][0]) * (aligned_points[i][1] - aligned_points[i-1][1])

        # Use the signed area and curvature to classify the edge
        if abs(curvature[0]) > threshold:
            if signed_area > 0:
                return +1  # Head
            else:
                return -1  # Hole
        else:
            return 0  # Flat
    
    def extract_colors(self, points):
        # points is a list of (x, y) tuples
        colors = []
        for point in points:
            x, y = point
            # Ensure the point is within the image dimensions
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                color = self.image[y, x]  # Access the color at the specified point
                colors.append(color)
            else:
                colors.append(None)  # Append None if the point is outside the image
        
        return cr.rgb2lab(np.array(colors))


    def match_edge(self, other, self_edge, other_edge):

        # Get the edge points of the two pieces
        self_type = self.edge_types[self_edge]
        other_type = other.edge_types[other_edge]
        if self_type == other_type:
            return False
        is_match = False
        if self_type != other_type and self_type != EdgeType.FLAT and other_type != EdgeType.FLAT:
            is_match = True
        
        return is_match


    
    # This method will draw and label edges on the image
    def draw_and_label_edges(self, threshold):
        # Find the indices of non-zero elements (edges)
        nonzero_indices = np.nonzero(self.canny_edge > 0)

        # Use boolean indexing to get the points
        edge_points = [(j, i) for i, j in zip(nonzero_indices[0], nonzero_indices[1])]
        corners = [(int(x), int(y)) for [x, y] in self.corners.reshape(-1, 2)]
        # Sort the edge points by proximity to the first corner
        edge_points = self.sort_points_clockwise(edge_points)
        # Convert edge points to a list of tuples
        corners = self.find_closest_points(corners, edge_points)
        #print(edge_points[corners])
        corners = self.sort_points_clockwise(corners)
        #utils.show_image(draw_points_on_image(self.image, corners))
        k = 0
        for i, corner in enumerate(corners):
            k +=1
            next_corner = corners[(i + 1) % len(corners)]
            # Find indices of the corners in the edge points
            start_index = edge_points.index(corner)
            end_index = edge_points.index(next_corner)
            # Extract points between corners
            if start_index < end_index:
                boundary_points = edge_points[start_index:end_index+1]
            else:  # Wrap around the end of the edge points
                boundary_points = edge_points[start_index:] + edge_points[:end_index+1]
            # Classify the edge
            self.boundaries.append(boundary_points)
            self.edges_colors = self.extract_colors(boundary_points)

            if k == 4:
                edge_type = self.classify_edge(boundary_points, threshold, -1)
            else:
                edge_type = self.classify_edge(boundary_points, threshold, 1)
            
            if edge_type == 0:
                self.edge_types.append(EdgeType.FLAT)
                edge_color = (255, 255, 255)
            elif edge_type == 1:
                self.edge_types.append(EdgeType.HEAD)
                edge_color = (0, 255, 0)
            else:
                self.edge_types.append(EdgeType.HOLE)
                edge_color = (0, 0, 255) 
                
            for point in boundary_points:
                cv2.circle(self.image, point, 1, edge_color, -1)
        
        flat_edges = []
        for type in self.edge_types:
            if type == EdgeType.FLAT:
                flat_edges.append(1)
        if len(flat_edges) == 0:
            self.piece_type = PieceType.MIDDLE
        if len(flat_edges) == 1:
            self.piece_type = PieceType.EDGE
        if len(flat_edges) == 2:
            self.piece_type = PieceType.CORNER
        #utils.show_image(self.image, 0.5)
    
    def get_canny_edge(self):
        mask = np.float32(self.mask).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assuming the largest contour is the puzzle piece
        puzzle_contour = max(contours, key=cv2.contourArea)
        # Create an empty mask for the puzzle contour
        puzzle_mask = np.zeros_like(mask)
        cv2.drawContours(puzzle_mask, [puzzle_contour], -1, (255), thickness=cv2.FILLED)
        edges = cv2.Canny(puzzle_mask, 100, 200)
        return edges
