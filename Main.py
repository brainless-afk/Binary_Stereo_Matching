import cv2
from matplotlib import pyplot as plt
import BinaryStereoMatching
import config  # config file
import time


def main():
    # load and display images
    img_left = cv2.imread(config.left_path)
    img_right = cv2.imread(config.right_path)
    img_ground = cv2.imread(config.true_disp_path)

    # Convert images from color to gray
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    bsm = BinaryStereoMatching.BinaryStereoMatching(img_left_gray, img_right_gray)

    print("Calculating disparity ...")
    t0 = time.process_time()
    disparity_map = bsm.calculate_disparity_map()
    t1 = time.process_time()
    print("Total: ", (t1 - t0))

    if config.show_input_images:
        cv2.imshow("Left Input Image", img_left)
        cv2.imshow("Right Input Image", img_right)
        if img_ground is not None:
            cv2.imshow("True Disparity", img_ground)

    if config.visualize_distribution:
        bsm.visualize_gaussian_distribution()

    if config.show_opencv_solution:
        opencv_solution(img_left, img_right)

    # normalize values in array to use full range of gray-scale image (0-255)
    cv2.normalize(disparity_map, disparity_map, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("Disparity", disparity_map)

    if config.save_result:
        filename = config.output_name + "_blockSize_" + str(config.blockSize) + "_n_" + str(config.n) + "_dmax_" + str(
            config.d_max) + ".png"
        cv2.imwrite(filename, disparity_map)

    print("Test: end of main")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Implementation with OpenCV functions
def opencv_solution(left_img, right_img):
    # Compute the disparity map using the stereo block matching algorithm
    stereo_bm = cv2.StereoBM_create(64, 9)
    dispmap_bm = stereo_bm.compute(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY))

    # Compute the disparity map using the stereo semi-global block matching algorithm
    stereo_sgbm = cv2.StereoSGBM_create(0, 64, 9)
    dispmap_sgbm = stereo_sgbm.compute(right_img, left_img)

    # Visualize the results
    plt.figure('OpenCV', figsize=(12, 10))
    plt.subplot(221)
    plt.title('left')
    plt.imshow(left_img[:, :, [2, 1, 0]])
    plt.subplot(222)
    plt.title('right')
    plt.imshow(right_img[:, :, [2, 1, 0]])
    plt.subplot(223)
    plt.title('BM')
    plt.imshow(dispmap_bm, cmap='gray')
    plt.subplot(224)
    plt.title('SGBM')
    plt.imshow(dispmap_sgbm, cmap='gray')
    plt.show()

    #plt.imsave("BM_low", dispmap_bm, cmap='gray')
    #plt.imsave("SGBM_low", dispmap_sgbm, cmap='gray')


if __name__ == "__main__":
    main()
