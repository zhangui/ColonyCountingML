import os
import cv2
import shutil

def get_path(i):

    script_dir = os.path.dirname(__file__)
    rel_path = "".join(["colony images/colony image (",str(i),").jpg"])
    abs_file_path = os.path.join(script_dir, rel_path)
    os.path.normpath(abs_file_path)

    return abs_file_path

def conv_to_binary(i):

    abs_file_path = get_path(i)
    img = cv2.imread(abs_file_path,2)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret, bw_img = cv2.threshold(blur, 100, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    ret, bw_img = cv2.threshold(bw_img, 75, 255, cv2.THRESH_BINARY)

    # cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Binary Image',cv2.resize(bw_img, (600, 600)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return bw_img


def save_binary(i,bw_img):
    script_dir = os.path.dirname(__file__)
    abs_dir_path = os.path.join(script_dir, "binary")

    rel_path = "".join(["colony image binary(",str(i),").jpg"])
    abs_file_path = os.path.join(abs_dir_path, rel_path)
    os.path.normpath(abs_file_path)
    cv2.imwrite(abs_file_path, bw_img)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    abs_dir_path = os.path.join(script_dir, "binary")

    # check if directory exists
    flag = os.path.isdir(abs_dir_path)
    if flag is True:
        shutil.rmtree(abs_dir_path)

    os.mkdir(abs_dir_path)

    for i in range(0,32):
        bw_img = conv_to_binary(i+1)
        save_binary(i+1,bw_img)