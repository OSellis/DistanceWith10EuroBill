# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:05:52 2017

@author: OSellis
"""

import numpy as np
import cv2

from matplotlib import pyplot as plt

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    gray = cv2.GaussianBlur(image, (7, 7), 0)
    
    edged = cv2.Canny(image, 150, 200, True, 5)
#    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
     
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
    _, contours, _= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
 
	# compute the bounding box of the of the paper region and return i
    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edged,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    return cv2.minAreaRect(c)

img = cv2.imread('20.jpg', 0)

image = find_marker(img)

#plt.subplot(121),plt.imshow(img)
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])


plt.show()
