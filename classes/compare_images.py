import cv2
import numpy as np

class CompareImage(object):
    
    def __init__(self, image1, image2):
        self.minimum_commutative_image_diff = 1
        self.image1 = image1
        self.image2 = image2
    
    def compare_image(self):
        image_1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        image_2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        commutative_image_diff = self.get_image_difference(image_1, image_2)
    
        if commutative_image_diff < self.minimum_commutative_image_diff:
            return commutative_image_diff
        return 10000 #random failure value
    
    def compare_images_features(self):
        image = self.image1
        template = self.image2

        detector = cv2.SIFT_create()        

        kp1, des1 = detector.detectAndCompute(image, None)
        kp2, des2 = detector.detectAndCompute(template, None)

        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        

            # Filter the good matches
            good = []
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    good.append([m])
        except:
            return False
        # Check if enough matches are found
        if len(good) > 4:
            print(True)
            return True
        return False
    
    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])
    
        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match
    
        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff/10) + img_template_diff
        return commutative_image_diff
