# Standard imports
import cv2
import numpy as np 
 
# Read images
src = cv2.imread("images/airplane.jpg")
dst = cv2.imread("images/sky.jpg")
 
# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
#poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
poly = np.array([
	[7, 796],
	[267, 643],
	[1452, 780],
	[2749, 430],
	[2620, 880],
	[2925, 822],
	[2652, 1056],
	[1017, 1309],
	[17, 996]
], np.int32)
cv2.fillPoly(src_mask, [poly], (255, 255, 255))
 
# This is where the CENTER of the airplane will be placed
center = (3072,2000)
 
# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
 
# Save result
cv2.imwrite("output/opencv-seamless-cloning-example.jpg", output)