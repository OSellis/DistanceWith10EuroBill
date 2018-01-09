# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:05:52 2017

@author: OSellis, SSchumann
"""

#------------------------------------------------
#   THE PROJECT
#------------------------------------------------
#   Measure distances using a 10 Euro bill.
#------------------------------------------------

# Constants, the width and height of the 10 Euro bill and the focal length of the camera.
# Got the focal point and distance formula from said source:
# https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

KNOWN_WIDTH = 125.0
KNOWN_HEIGHT = 67.0
FOCAL = 3420.8

#import numpy as np
import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

# The function which calculates the distance to the camera using the real life width of the bill, the focal length of the camera and the measured width from the image.

def distance_to_camera(knownWidth, focalLength, fromPicWidth):
    return (knownWidth * focalLength) / fromPicWidth

# The function to locate the edges of the 10 Euro bill.

def find_marker(image, low_thresh, high_thresh):
	
    # Blur the image to make the edges more uniform and detect the edges using the Canny function
    # The Canny is a basically uses the Sobel method both vertically and horizontally and checks if the edges are actually edges via non-maximum suppression and hystheresis thresholding.
    # More info from here: https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    
    filtered_contours = []
    gray = cv2.GaussianBlur(image, (7, 7), 0)    
    
    edged = cv2.Canny(gray, low_thresh, high_thresh, True, 3)
    
    # The edges are also diluted and eroded to make them more uniform.
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)
    
     
	# find the contours in the edged image and keep the 5 largest ones
	# we assume that our 10 Euro image is one of them. Also this sorts out contours that are detected via noise.
    im2, contours, _= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:5]
    

    # This part is written for the checking procedure, it draws rectangles around the countours, so that we could analyise if the algorithm got the 10 Euro bill.

    Height, Width = edged.shape
    min_x, min_y = Width,Height
    max_x = max_y = 0
    
    for contour in contours:

        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)

    # The bounding rectangle must be larger than 300 for both width and height, because otherwise it is also most likely noise.        
        
        if w > 300 and h > 300:
            
            cv2.rectangle(gray, (x,y), (max_x, max_y), (0, 0, 0), 2)
            filtered_contours.append(contour)
    
    # Return all contours that are bigger than 300 in height and width and also belong to the 5 biggest contours found.
    
    return filtered_contours, edged


# Read in the image and get the filtered contours

img = cv2.imread('test7.jpg')

# Read in template for 10
edges_10 = cv2.imread('template_edges.jpg',0)
edges_10_height, edges_10_width = edges_10.shape

# Three different methods for template matching, used in the following function
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED']


# The following function attempts to do template matching on the image
# and determines whether the number 10 is at a correct location wrt the rectangle discovered.
# This method rules out all rectangles that do not have number "10" on the correct
# location in them, meaning they are not the 10-euro bill.
def match_template(box):
    # Get the coordinates of the box and check for the ratio of sides
    (x,y,width,height) = box
    if abs(float(width)/height - 125.0/67.0) > 0.5:
        # This is the easiest way to eliminate a rectangle found:
        # the height-to-width ratio is incorrect.
        return False
    
    # Now we have determined that the rectangle has approximately correct height-to-width ratio.
    # Resize the template to match width.
    template = cv2.resize(edges_10,(edges_10_width*width/4128,edges_10_height*width/4128))
    w, h = template.shape[::-1]
    
    # Detect edges on the grayscale image using Canny edge detection, as earlier
    gray = cv2.blur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), (7, 7))
    edged = cv2.Canny(gray, 40, 100)
    edged = cv2.resize(edged,(edged.shape[1]/4,edged.shape[0]/4))
    
    # Try to apply each template matching method to the image
    # This code was largely inspired by OpenCV sample template matching code
    
    # Meanwhile, also keep track of how many times template matching has given us correct
    # coordinates for the match with respect to the rectangle discovered
    correctness_counter = 0
    
    for m in methods:
        # Make copy of the edge image, in order not to mess up the original edge image
        edged_copy = edged.copy()
        method = eval(m)
    
        # Apply template matching using method m
        res = cv2.matchTemplate(edged_copy,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        # Take maximum of the result as the location of the top left corner of the template
        top_left = max_loc
        # Calculate location of the bottom right corner
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # Now check whether the midpoint of the template falls on a reasonable place
        # inside the rectangle
        # Compare the location against a threshold
        threshold = 0.2
        # We compare the midpoint of the discovered template location
        # and the location on the 10-Euro bill we know the template to be from
        w_diff = abs((top_left[0] + w/2) - (x + width/2)/4)/float(width)
        h_diff = abs((top_left[1] + h/2) - (y + height/5.5)/4)/float(height)
        
        if w_diff < threshold and h_diff < threshold:
            # If template has proven to be matched to the correct location
            # with respect to the discovered rectangle, increment the correctness counter.
            correctness_counter += 1
        
    # Return the number of times the template was matched to the correct location
    # inside this rectangle. Maximum number 3, minimum 0.
    return correctness_counter


# Initialise variables correct_box and correctness_level so we can iterate over
# a range of layers and thresholds to find the best match for the bill from the image
correct_box = None
correctness_level = 0

# Now test the image in the following way.
# In each of the R, G and B layers, attempt to determine the rectangles that could represent
# the 10-Euro bill using 5 different thresholds (this could ideally be replaced by a range
# of different thresholds instead of just individual values, but then running the code
# takes a lot longer).
# For each of the rectangles found on the image for which we suspect it could be the bill,
# run the template matching function above, which also checks for width-to-height ratio.
# Save the correctness level and if the discovered rectangle has a higher correctness level
# than any of the ones discovered earlier, save it for future use.
for layer in range(3):
    for low_thresh, high_thresh in [(30,130),(40,100),(5,90),(0,30),(20,22)]:
        # Find rectangles using specified layer and thresholds
        Areas, edged = find_marker(img[:,:,layer],low_thresh,high_thresh)
        
        # Create bounding boxes for the contours discovered and sort them
        boundingBoxes = [cv2.boundingRect(a) for a in Areas]
        if not boundingBoxes:
            continue
        (Areas, boundingBoxes) = zip(*sorted(zip(Areas, boundingBoxes), key=lambda b:b[1][0], reverse=True))
        
        # Check each bounding box against the template
        # Save box in variable correct_box if necessary
        for box in boundingBoxes:
            correctness = match_template(box)
            if correctness > correctness_level:
                correctness_level = correctness
                correct_box = copy.deepcopy(box)


# There is a chance there is no 10-Euro bill on the image
# or that the 10-Euro bill is not detected correctly by the algorithm.
# If there are not correct matches, print a corresponding message
# and exit the program.
if not correctness_level:
    print "no correct rectangle found"
    exit()

(x,y,width,height) = correct_box
#   Present the results in an image for us to check the thresholding and results of edge detection.
plt.subplot(122),plt.imshow(edged,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# Draw the smallest bounding box to the picture and present the results
cv2.rectangle(img, (x,y), (x+width, y+height), (255, 0, 0), 10)

# Because pyplot operates on RGB images, but ours is a BGR, convert it for visualisation purposes
img2 = img.copy()
img2[:,:,0] = img[:,:,2]
img2[:,:,2] = img[:,:,0]

# Display the original image that also has the box drawn around the 10-Euro bill found
plt.subplot(121),plt.imshow(img2)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# Calculate distance to the camera
distance = distance_to_camera(KNOWN_WIDTH, FOCAL, width)

# Display results to the user
print("Distance to camera: "+str(distance)+"cm")

plt.show()
