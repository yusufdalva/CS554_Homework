import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class DisparityMapEstimator:

    def __init__(self, max_disparity=65, color_mode="gray", depth_mode="window"):
        self.right_image = None
        self.left_image = None
        self.max_disparity = max_disparity
        if color_mode not in ("rgb", "gray"):
            raise AttributeError("Invalid color mode: It should be 'gray' for grayscale and 'rgb' for RGB colors")
        self.color_mode = "gray"
        if depth_mode not in ("window", "pixel"):
            raise AttributeError("Depth value (depth_mode) can only be determined window-based('window') or center-based('center')")
        self.depth_mode = depth_mode

    """ This method is added to enable running the implementation with different image pairs, in a rapid way.
    The method changes the current images for the right and left views. """
    def register_images(self, left_path, right_path):
        if not os.path.isfile(right_path):
            raise AttributeError("Path for the right view is invalid. Given path: {}".format(right_path))
        if not os.path.isfile(left_path):
            raise AttributeError("Path for the left view is invalid. Given path: {}".format(left_path))
        
        # Read images
        if self.color_mode == "rgb":
            self.left_image = cv2.imread(left_path)
        elif self.color_mode == "gray":
            self.left_image = cv2.imread(left_path, 0)
        if self.color_mode == "rgb":
            self.right_image = cv2.imread(right_path)
        elif self.color_mode == "gray":
            self.right_image = cv2.imread(right_path, 0)
        
        if self.right_image.shape != self.left_image.shape:
            raise ValueError("Selected right and left images are not equal in size")

        if self.right_image is not None:
            print("INFO: Right view image loaded. The shape of the image is {}".format(self.right_image.shape))
        if self.left_image is not None:
            print("INFO: Left view image loaded. The shape of the image is {}".format(self.left_image.shape))


    def normalize_rgb_patch(self, image_patch):
        # Step 1: Find the mean pixel value
        avg_pixel = np.mean(image_patch, axis=(0,1)) # Find average RGB pixel, mean over x and y coords

        # Step 2: Find magnitude of [img pixels - average pixel]
        zero_centered_img = image_patch - avg_pixel

        # Step 3: Normalizing the patch
        img_norm = np.sqrt(np.sum(np.square(zero_centered_img), axis=(0,1)))
        # Handling zero norm
        if img_norm[0] == 0.0:
            img_norm[0] = 1e-7
        if img_norm[1] == 0.0:
            img_norm[1] = 1e-7
        if img_norm[2] == 0.0:
            img_norm[2] = 1e-7
        
        return zero_centered_img / img_norm

    
    def normalize_gray_patch(self, image_patch):
        # Step 1: Find the mean pixel value
        avg_pixel = np.mean(image_patch)

        # Step 2: Find magnitude of [img pixels - average pixel]
        zero_centered_img = image_patch - avg_pixel

        # Step 3: Normalizing the patch
        img_norm = np.sqrt(np.sum(np.square(zero_centered_img)))
        # Handling zero norm
        if img_norm == 0.0:
            img_norm = 1e-7
    
        return zero_centered_img / img_norm

    
    def normalize_patch(self, image_patch):
        if self.color_mode == "rgb":
            return self.normalize_rgb_patch(image_patch)
        return self.normalize_gray_patch(image_patch)

    
    def unwrap_image_patch(self, image_patch):
        patch_rows = []
        for patch_row in range(image_patch.shape[0]):
            if patch_row % 2 == 0:
                patch_rows.append(image_patch[patch_row])
            else:
                patch_rows.append(np.flipud(image_patch[patch_row]))
        return np.concatenate(patch_rows)


    def process_patch_pairs_rgb(self, left_patch, right_patch):
        unwrapped_left = np.reshape(left_patch, (left_patch.shape[0] * left_patch.shape[1], left_patch.shape[2]))
        unwrapped_right = np.reshape(right_patch, (right_patch.shape[0] * right_patch.shape[1], right_patch.shape[2]))
        total_distance = np.sum(np.linalg.norm(left_patch - right_patch, axis=-1))
        return total_distance

    def process_patch_pairs_gray(self, left_patch, right_patch):
        normalized_left = self.normalize_patch(left_patch)
        normalized_right = self.normalize_patch(right_patch)
        #unwrapped_left = self.unwrap_image_patch(normalized_left)
        #unwrapped_right = self.unwrap_image_patch(normalized_right)
        unwrapped_left = normalized_left.flatten()
        unwrapped_right = normalized_right.flatten()
        cos_sim = np.dot(unwrapped_left, unwrapped_right)
        return cos_sim

    def process_patch_pairs(self, left_patch, right_patch):
        if self.color_mode == "rgb":
            return self.process_patch_pairs_rgb(left_patch, right_patch)
        return self.process_patch_pairs_gray(left_patch, right_patch)


    def get_map_for_rgb(self, window_width):
        # Window size needs to be odd
        if window_width % 2 == 0:
            window_width += 1
        window_margin = window_width // 2
        disparity_map = np.zeros((self.left_image.shape[0], self.left_image.shape[1]))
        for row_idx in tqdm(range(window_margin, self.left_image.shape[0] - window_margin)):
            for col_idx in range(window_margin, self.left_image.shape[1] - window_margin):
                min_dist = float("inf")
                disp_value = 0
                left_patch = self.left_image[row_idx - window_margin: row_idx + window_margin + 1, col_idx - window_margin: col_idx + window_margin + 1]
                norm_left_patch = self.normalize_patch(left_patch)
                disp_limit = self.left_image.shape[1] - col_idx - window_margin
                for disparity in range(min(self.max_disparity + 1, disp_limit)):
                    right_patch = self.right_image[row_idx - window_margin: row_idx + window_margin + 1, col_idx + disparity - window_margin: col_idx + disparity + window_margin + 1]
                    norm_right_patch = self.normalize_patch(right_patch)
                    distance = self.process_patch_pairs(norm_left_patch, norm_right_patch)
                    if distance < min_dist:
                        min_dist = distance
                        disp_value = disparity
                if self.depth_mode == "window":
                    disparity_window = disp_value * np.ones((window_width, window_width))
                    disparity_map[row_idx - window_margin: row_idx + window_margin + 1, col_idx - window_margin: col_idx + window_margin + 1] += disparity_window
                else:
                    disparity_map[row_idx, col_idx] = disp_value
        return disparity_map


    def get_map_for_gray(self, window_width):
        # Window size needs to be odd
        if window_width % 2 == 0:
            window_width += 1
        window_margin = window_width // 2
        disparity_map = np.zeros((self.left_image.shape[0], self.left_image.shape[1]))
        for row_idx in tqdm(range(window_margin, self.left_image.shape[0] - window_margin)):
            for col_idx in range(window_margin, self.left_image.shape[1] - window_margin):
                max_sim = float("-inf")
                disp_value = 0
                left_patch = self.left_image[row_idx - window_margin: row_idx + window_margin + 1, col_idx - window_margin: col_idx + window_margin + 1]
                disp_limit = self.left_image.shape[1] - col_idx - window_margin
                for disparity in range(min(self.max_disparity + 1, disp_limit)):
                    right_patch = self.right_image[row_idx - window_margin: row_idx + window_margin + 1, col_idx + disparity - window_margin: col_idx + disparity + window_margin + 1]
                    cos_sim = self.process_patch_pairs(left_patch, right_patch)
                    if cos_sim > max_sim:
                        max_sim = cos_sim
                        disp_value = disparity
                if self.depth_mode == "window":
                    disparity_window = disp_value * np.ones((window_width, window_width))
                    disparity_map[row_idx - window_margin: row_idx + window_margin + 1, col_idx - window_margin: col_idx + window_margin + 1] += disparity_window
                else:
                    disparity_map[row_idx, col_idx] = disp_value
        return disparity_map

    
    def get_map(self, window_width):
        if self.color_mode == "rgb":
            return self.get_map_for_rgb(window_width)
        return self.get_map_for_gray(window_width)



if __name__ == "__main__":
    print("INFO: Starting test")
    estimator = DisparityMapEstimator()
    data_path = os.path.join(os.getcwd(), "homework_dataset/data_disparity_estimation")
    right_path = os.path.join(data_path, "plastic/right.png")
    left_path = os.path.join(data_path, "plastic/left.png")
    estimator.register_images(right_path, left_path)
    d_map = estimator.get_map(5)
    plt.imshow(d_map / np.max(d_map), cmap="gray")
    plt.show()
    

