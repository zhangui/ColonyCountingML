import numpy as np
from PIL import Image

img = Image.open('testrgba.png')
array = np.array(img)
print(array.shape)      # (100, 200, 4)
