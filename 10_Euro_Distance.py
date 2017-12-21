# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:05:52 2017

@author: OSellis
"""

#------------------------------------------------
#   THE PROJECT
#------------------------------------------------
#   Measure distances using a 10 Euro bill.
#------------------------------------------------

# Constants, the width and height of the 10 Euro bill and the focal length of the camera.
# Got the focal point and distance formula from said source:
# https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

DISTANCE_FROM_CAMERA = 400
KNOWN_WIDTH = 125.0
KNOWN_HEIGHT = 67.0
FOCAL = 3420.8

#import numpy as np
import cv2

from matplotlib import pyplot as plt

# The function which calculates the distance to the camera using the real life width of the bill, the focal length of the camera and the measured width from the image.

def distance_to_camera(knownWidth, focalLength, fromPicWidth):
    return (knownWidth * focalLength) / fromPicWidth

# The function to locate the edges of the 10 Euro bill.

def find_marker(image):
	
#    Blur the image to make the edges more uniform and detect the edges using the Canny function
#    The Canny is a basically uses the Sobel method both vertically and horizontally and checks if the edges are actually edges via non-maximum suppression and hystheresis thresholding.
#    More info from here: https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html

    
    filtered_contours = []
    gray = cv2.GaussianBlur(image, (7, 7), 0)    
    
    edged = cv2.Canny(gray, 40, 100, True, 3)
    
#    The edges are also diluted and eroded to make them more uniform.
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)
     
	# find the contours in the edged image and keep the 5 largest ones
	# we assume that our 10 Euro image is one of them. Also this sorts out contours that are detected via noise.
    im2, contours, _= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:5]
    

#   This part is written for the checking procedure, it draws rectangles around the countours, so that we could analyise if the algorithm got the 10 Euro bill.

    Height, Width = edged.shape
    min_x, min_y = Width,Height
    max_x = max_y = 0
    
    for contour in contours:

        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)


#   The bounding rectangle must be larger than 300 for both width and height, because otherwise it is also most likely noise.        
        
        if w > 300 and h > 300:
            
            cv2.rectangle(gray, (x,y), (max_x, max_y), (0, 0, 0), 2)
            filtered_contours.append(contour)
        
        
#   Present the results in an image for us to check the thresholding and results of edge detection.
            

    plt.subplot(122),plt.imshow(edged,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
#   Return all contours that are bigger than 300 in height and width and also belong to the 5 biggest contours found.
    
    return filtered_contours

#   Read in the image and get the filtered contours

img = cv2.imread('30.jpg')

Areas = find_marker(img[:,:,1])

#  Sort the areas and take the smallest of these contours, because visual analysis showed that other contours are bigger than the 10 Euro bill.

boundingBoxes = [cv2.boundingRect(a) for a in Areas]
(Areas, boundingBoxes) = zip(*sorted(zip(Areas, boundingBoxes), key=lambda b:b[1][0], reverse=True))

(x,y,width,height) = boundingBoxes[0]

#   Draw the smallest bounding box to the picture and present the results

cv2.rectangle(img, (x,y), (x+width, y+height), (255, 0, 0), 2)

plt.subplot(121),plt.imshow(img)
plt.title('Gray Image'), plt.xticks([]), plt.yticks([])


# Algorithm to find the focal point of the camera.

#focal = (width * DISTANCE_FROM_CAMERA) / KNOWN_WIDTH
#print(focal)

distance = distance_to_camera(KNOWN_WIDTH, FOCAL, width)

print(distance)


plt.show()
