# from scipy.cluster.vq import vq, kmeans, whiten
# import sklearn
from scipy.cluster.vq import kmeans2,kmeans
import matplotlib.pyplot as plt
import cv2
import numpy as np


def getloss(j,points):
	centroid, label = kmeans2(points.astype(np.float32), j*10, minit='points')
	curr_sse = 0
	for i in range(len(points)):
		
		curr_center = centroid[label[i]]
		curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
	return curr_sse

def kmeans(img,max_iteration = 150):
	img = cv2.resize(img, (400,400)) 
	img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
	n_white_pix = np.sum(img == 255)
	#print("white pix: ", n_white_pix)
	n_black_pix = np.sum(img == 0)
	#print("black pix: ", n_black_pix)
	if min(n_white_pix,n_black_pix) < max_iteration*10:
		max_iteration = min(n_black_pix,n_white_pix)//10
	if n_white_pix > n_black_pix: 
		cv2.bitwise_not(img, img)
	points = np.argwhere(img == 255)

	sse = []
	min_sse = getloss(1,points)
	prev_sse = min_sse
	for i in range(10):
		for j in range(1, max_iteration):
			curr_sse = getloss(j,points)
			normalized_sse = np.abs((prev_sse - curr_sse))
			if (normalized_sse<200000 and normalized_sse/prev_sse < 0.1):
				sse.append(j)
				break
			prev_sse = curr_sse
	#print("Final result: ",np.mean(sse))
	return np.mean(sse)



for i in range(32):
	img = cv2.imread('binary_masked_resized/colony image resized(' + str(i+1) + ').jpg', 0) # read image
	result = kmeans(img)
	print("The result for image "+str(i+1)+" is ", result)