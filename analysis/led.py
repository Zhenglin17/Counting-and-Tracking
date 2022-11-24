import os
import cv2
from scpye.utils.drawing import imshow

image_dir = "/home/chao/Pictures/led_test"


image_file = os.path.join(image_dir, 'ricoh_e2_g2_led_3_up_adjusted2.png')

image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (5, 5), 1)
imshow(image, figsize=(12, 16))
