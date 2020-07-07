import numpy as np
import os
from PIL import Image

script_dir = os.path.dirname(__file__)
rel_path = "colony images/colony image (1).jpg"
abs_file_path = os.path.join(script_dir, rel_path).replace("/","\\")
os.path.normpath(abs_file_path)

img = Image.open(abs_file_path)
array = np.array(img)
print(array.shape)      # (1415, 1112, 3)
