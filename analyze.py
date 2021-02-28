import cv2
import os
from PIL import Image

image = cv2.imread("images/center_2016_12_01_13_30_48_287.jpg")
print(image)
# mean, stdDev = cv2.meanStdDev(img)
# z-score normalization
# image -= image.min()
# image /= image.max()

# image *= 255 # [0, 255] range
stand_img = (image / 255) - 0.5

print(stand_img)
cv2.imwrite("images/norm_center.jpg", stand_img*255)

# ((70, 25), (0, 0))
original = Image.open("images/center_2016_12_01_13_30_48_287.jpg")
width, height = original.size

cropped_example = original.crop((0, 70, width, height-25))
cropped_example.save("images/cropped_center.jpg")

# Flipping image
flipped_image = cv2.flip(image, 1)
cv2.imwrite("images/flipped_center.jpg", flipped_image)
