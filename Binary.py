import numpy as np
import os
import cv2

def get_path(i):

    script_dir = os.path.dirname(__file__)
    rel_path = "".join(["colony images/colony image (",str(i),").jpg"])
    abs_file_path = os.path.join(script_dir, rel_path)#.replace("/","\\")
    os.path.normpath(abs_file_path)

    return abs_file_path

def conv_to_binary():

    abs_file_path = get_path(8)
    img = cv2.imread(abs_file_path,2)
    print(img.size)
    ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    cv2.imshow('Binary Image',bw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

conv_to_binary()