import cv2
import numpy as np

def printImage(image):
    cv2.imshow('Test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Read in image, convert to grayscale and threshold
img = cv2.imread('image085447.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
printImage(thresh)
# Remove Noise
# MORPH_OPEN removes small 'white noise'
# MORPH_CLOSE removes holes, cleans edges
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Get confirmed BG area
sureBG = cv2.dilate(opening, kernel, iterations=3)
# Get confirmed FG area
distTransform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sureFG = cv2.threshold(distTransform, 0.7*distTransform.max(), 255, 0)
# Find unknown region (subtract FG from BG)
sureFG = np.uint8(sureFG)
unknown = cv2.subtract(sureBG, sureFG)

# Marker labelling
ret, markers = cv2.connectedComponents(sureFG)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
printImage(img)