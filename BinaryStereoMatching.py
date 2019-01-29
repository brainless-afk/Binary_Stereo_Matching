# Created by Felix N. at 14.01.2019

import numpy as np
from matplotlib import pyplot as plt
import config


class BinaryStereoMatching:

    def __init__(self, image_left, image_right):
        self.image_left = image_left
        self.image_right = image_right

        self.blockSize = config.blockSize
        self.offset = int(self.blockSize / 2)  # because x is in the middle of the template
        self.pair_number = config.n

        # gaussian distribution arrays for image pairs in S*S window
        # split 2D coordinates array into two arrays for better performance when accessing arrays later
        gauss_pts_p = self.gauss_distribution()
        gauss_pts_p = np.around(gauss_pts_p)
        self.pts_p_x, self.pts_p_y = gauss_pts_p.astype(int).T

        gauss_pts_q = self.gauss_distribution()
        gauss_pts_q = np.around(gauss_pts_q)
        self.pts_q_x, self.pts_q_y = gauss_pts_q.astype(int).T

    def calculate_disparity_map(self):
        image_left = self.image_left
        image_right = self.image_right
        if image_left.shape != image_right.shape:
            print("Error: Size of images doesn't match")
            return
        binary = "1"
        offset = self.offset
        pair_number = self.pair_number + 1
        height, width = image_left.shape

        d_min = config.d_min
        d_max = config.d_max
        tmp_best_c = 1 << pair_number  # just a higher than possible starting number
        disparity_map = np.zeros(image_left.shape, np.uint8)

        for y in range(offset, height - offset):
            descriptors_left = self.calculate_row_descriptors(image_left, width, y)
            descriptors_right = self.calculate_row_descriptors(image_right, width, y)
            descriptor_list_length = len(descriptors_right)

            for pos, b_x_left in enumerate(descriptors_left, start=0):
                best_c_x_d = tmp_best_c
                d_x = 0

                # check for best C(x,d) in disparity range
                for d in range(d_min, d_max):
                    # check if out of bounds of list length
                    if d + pos >= descriptor_list_length:
                        break
                    b_x_right = descriptors_right[d + pos]

                    # "^" yields the bitwise XOR operator for integer arguments
                    c_x_d = bin(b_x_left ^ b_x_right).count(binary)  # hamming distance

                    if c_x_d < best_c_x_d:
                        d_x = d
                        best_c_x_d = c_x_d

                disparity_map[y, pos + offset] = d_x

        return disparity_map

    # calculate all B(x) for a row in image
    def calculate_row_descriptors(self, image, width, y_pos):
        # local variables are faster than class member access
        offset = self.offset
        pair_number = self.pair_number - 1
        pts_p_x = self.pts_p_x
        pts_p_y = self.pts_p_y
        pts_q_x = self.pts_q_x
        pts_q_y = self.pts_q_y
        descriptor_list = []

        for x in range(offset, width - offset):
            # calculate descriptor with n pairs of pixels
            descriptor = 0

            for index in range(pair_number):
                p_i = image.item(y_pos + pts_p_y.item(index), x + pts_p_x.item(index))
                q_i = image.item(y_pos + pts_q_y.item(index), x + pts_q_x.item(index))

                # add 2^index to sum if true
                if p_i > q_i:
                    descriptor += 1 << index  # 1 shifted i positions to the left equals 2^i
            descriptor_list.append(descriptor)

        return descriptor_list

    def gauss_distribution(self):
        pts = np.random.multivariate_normal(mean=[0, 0], cov=[[self.offset, 0], [0, self.offset]], size=config.n,
                                            check_valid='warn')

        # values outside of accepted range are clipped to respective min or max value
        return np.clip(pts, -self.offset, self.offset)

    # visualizes the used distribution algorithm
    def visualize_gaussian_distribution(self):
        pts = self.gauss_distribution()

        plt.figure('Gaussian Distribution', figsize=(15, 7))

        plt.subplot(121)
        plt.title('Histogram')
        plt.hist(pts, bins=config.bins)

        plt.subplot(122)
        plt.title('Coordinates')
        plt.scatter(pts[:, 0], pts[:, 1], s=1)
        plt.xlim((-self.offset, self.offset))
        plt.ylim((-self.offset, self.offset))
        plt.show()
