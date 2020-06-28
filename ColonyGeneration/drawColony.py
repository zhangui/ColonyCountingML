from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

# variables for generating colonies play around to get desired result or add random 
colony_count = 100 #np.random.randint(0, 100)
radius_sigma = 1
radius_mu = 8


for i in range(0, colony_count):
    colony_radius = np.random.normal(radius_mu, radius_sigma) # radius is gaussian normal distribution
    colony_x = np.random.randint(margin_width, xsize - margin_width)
    colony_y = np.random.randint(margin_width, ysize - margin_width)
    while (circle_contains(petri_centerx, petri_centery, petri_radius, colony_x, colony_y, colony_radius) == False):
            colony_radius = np.random.normal(radius_mu, radius_sigma)
            colony_x = np.random.randint(margin_width, xsize - margin_width)
            colony_y = np.random.randint(margin_width, ysize - margin_width)
    x1 = colony_x - colony_radius
    y1 = colony_y - colony_radius
    x2 = colony_x + colony_radius
    y2 = colony_y + colony_radius
    draw.ellipse((x1, y1, x2, y2), fill = 'black', outline ='black')
 

img.save('colony1.png')