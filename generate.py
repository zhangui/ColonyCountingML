# TODO: change ellipse position to be random (within boundaries)
# TODO: add a random 3d change-perspective operation

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import os
import sys
import csv

import shutil

# Arguments Format:
#     python.py generate.py `path to generate data at` `number of images for training` `number of images for validation`

num_of_param = len(sys.argv) - 1
if num_of_param < 3:
    print('Error: command requires 3 arguments: save directory, number of images for training, and number of images for validation')
    exit()

def make_filename(basedir, curdir, basename, index, extension):
    filename = basename + str(index) + '.' + extension
    return os.path.join(basedir, curdir, filename)

def circle_contains(bigx, bigy, bigrad, smallx, smally, smallrad):
    d = np.sqrt((bigx - smallx)**2 + (bigy - smally)**2)
    if (bigrad > smallrad + d):
        return True
    return False

xsize = 1520
ysize = 1520
edge_width = 30
margin_width = 20
img = Image.new('RGB', (xsize, ysize), color = (255, 255, 255))
petri_radius = (xsize - 2 * margin_width)/2
petri_centerx = xsize / 2
petri_centery = ysize / 2

draw = ImageDraw.Draw(img)
draw.ellipse((margin_width, margin_width, xsize - margin_width, ysize - margin_width), fill = 'white', outline ='black')
draw.ellipse((margin_width + edge_width, margin_width + edge_width, xsize - margin_width - edge_width, ysize - margin_width - edge_width), fill = 'white', outline ='black')


base_dir = sys.argv[1]

if not sys.argv[2].isdigit():
    print('Error: Second argument must be an integer')
    exit()

if not sys.argv[3].isdigit():
    print('Error: Second argument must be an integer')
    exit()

dataset_size = [int(sys.argv[2]), int(sys.argv[3])]

cur_dirs = ['data/train', 'data/validate']
file_basename = 'colony'
file_ext = 'png'

if os.path.exists(os.path.join(base_dir, 'data')):
    response = raw_input('The path given already exists. Would you like to delete it? [y/n] ')
    if len(response) != 0 and response[0] == 'y':
        shutil.rmtree(os.path.join(base_dir, 'data'))
    else:
        print('Data generation failed. Please choose a different directory or delete the given one')
        exit()


# Populate train and validate directories
for dataset_size_index, cur_dir in enumerate(cur_dirs, 0):
    os.makedirs(os.path.join(base_dir, cur_dir))

    # Store each count in a list and then write to avoid possible overwritting.
    # This is because image creation time may be large, during which one might change
    # 'labels.csv'.
    colony_counts = []

    print('Populating ' + cur_dir + '...')
    for index in range(dataset_size[dataset_size_index]):
        colony_count = np.random.randint(1, 100)
        colony_counts.append(colony_count)

        radius_sigma = 1
        radius_mu = 8


        for i in range(0, colony_count):
            colony_radius = np.random.normal(radius_mu, radius_sigma) # radius is normally distribution
            colony_x = np.random.randint(margin_width, xsize - margin_width)
            colony_y = np.random.randint(margin_width, ysize - margin_width)
            while (circle_contains(petri_centerx, petri_centery, petri_radius,
                                   colony_x, colony_y, colony_radius) == False):
                colony_radius = np.random.normal(radius_mu, radius_sigma)
                colony_x = np.random.randint(margin_width, xsize - margin_width)
                colony_y = np.random.randint(margin_width, ysize - margin_width)
            x1 = colony_x - colony_radius
            y1 = colony_y - colony_radius
            x2 = colony_x + colony_radius
            y2 = colony_y + colony_radius
            draw.ellipse((x1, y1, x2, y2), fill = 'black', outline ='black')
    

        # Save image
        img.save(make_filename(base_dir, cur_dir, file_basename, index, file_ext))


    # Save labels for basedir
    with open(os.path.join(base_dir, cur_dir, 'labels.csv'), 'wb') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i, cur_count in enumerate(colony_counts, 0):
            file_path  = os.path.abspath(make_filename(base_dir, cur_dir, file_basename, i, file_ext))
            writer.writerow([file_path, cur_count])

print('Data is located at "' + os.path.abspath(os.path.join(base_dir, 'data')) + '"')
