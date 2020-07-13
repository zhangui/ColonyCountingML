import cv2
import numpy as np

img = cv2.imread('colony_image_binary_8_.jpg', 0) # read image
 
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

height, width = img.shape[:2]
total_pixels = height * width
mask_white_pixel_thresh = total_pixels * 500 / 360000 # min number of white pixels changes with image size

n_white_pix = np.sum(img == 255)
n_black_pix = np.sum(img == 0)

if n_white_pix > n_black_pix: # if original image is mainly white
    cv2.bitwise_not(img, img)

finalimage = img.copy()

num_labels, labels = cv2.connectedComponents(img) # find connected component
mask = np.array(labels, dtype=np.uint8)
for label in range(1,num_labels):
    new_mask = np.array(labels, dtype=np.uint8)
    
    
    new_mask[labels == label] = 255 
    mask_white_pix = np.sum(new_mask == 255)
    if mask_white_pix > mask_white_pixel_thresh:
        mask[labels == label] = 255
    else:
        mask[labels == label] = 0

cv2.bitwise_not(finalimage, finalimage, mask)

if n_white_pix > n_black_pix:
    cv2.bitwise_not(finalimage, finalimage) 


filename = 'petriremoved.jpg'
cv2.imwrite(filename, finalimage)

