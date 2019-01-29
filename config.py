# === Configuration file ===
# stores variables used in code for easy modification

# === Image Paths ===
# Left Image path
left_path = 'Images/tsukuba/scene1.row3.col3.ppm'

# Right Image path
right_path = 'Images/tsukuba/scene1.row3.col1.ppm'

# True Disparity Image path
# to compare this programs output to a true disparity image if one exists
true_disp_path = 'Images/tsukuba/truedisp.row3.col3.pgm'

# === Variables ===
# Size of the S * S window centered on pixel x
# size should be odd as as the block is centered around a pixel
blockSize = 27

# n = pairs of pixels calculated by gaussian distribution
# used to calculate BRIEF descriptor B(x)
n = 128

# Disparity ranges of Dd, in most cases Dd = [0, d_max -1]
d_min = 0
d_max = 64

# == Visualization ===
# Display images specified in "Image Paths"
show_input_images = False

# Visualizes one example of data created by the gaussian distribution algorithm used in this program
visualize_distribution = False
# bins represents the number of columns shown for x and y values in the histogram
bins = 50

# shows depth map from StereoMatcher class from OpenCV
show_opencv_solution = True

# save result as image
save_result = False
output_name = "Disparity_"  # + blockSize value, n value, d_max
