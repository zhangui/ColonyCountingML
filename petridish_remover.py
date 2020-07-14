import cv2
import numpy as np

num_of_param = len(sys.argv) - 1
if num_of_param < 1:
    print('Error: command requires 1 arguments: the directory of images')
    exit()

print('Preparing preprocess...')
directory = sys.argv[1]
for filename in os.listdir(directory):    
    filename = os.path.join(directory, filename)
    img = None

    # If not an image file, then don't read it
    if not os.path.isfile(filename):
        pass
    
    try:
        img = cv2.imread(filename, 0) # read image
    except cv2.error as e:
        pass

    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    height, width = img.shape[:2]
    total_pixels = height * width

    # min number of white pixels changes with image size
    mask_white_pixel_thresh = total_pixels * 500 / 360000 

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

    cv2.imwrite(filename, finalimage)

print('Preprocess finished')
