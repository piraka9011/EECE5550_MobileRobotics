#!/usr/bin/python
"""Segment an image from a ROS topic based on LaB color scheme.

Requires OpenCV3.
"""
__author__ = "Anas Abou Allaban"
__maintainer__ = "Anas Abou Allaban"
__email__ = "anas@abouallaban.info"

# ROS
import rospy
import cv2
from collections import OrderedDict
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# Sys
import numpy as np
from math import sqrt, pow


class ColorSegmenter:
    def __init__(self):
        # Add colors as needed
        colors = OrderedDict({
            "red":      (255, 50, 10),
            "green":    (100, 255, 50),
            "blue":     (100, 120, 255),
            "yellow":   (255, 255, 0),
            "black":    (0, 0, 0)})

        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # Update the L*a*b* array and the color names list
        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)
        # Convert RGB to LAB
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def euclidean(self, x, y):
        # Use .tolist() or .astype(np.int64), otherwise overflow error
        x = x.tolist()
        y = y.tolist()
        return sqrt(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2) + pow(x[2] - y[2], 2))

    def getColorLabel(self, color):
        # Init. Min. Dist.
        minDist = (np.inf, None)

        # Loop over the known L*a*b* color values
        for i, row in enumerate(self.lab):
            # Compute distance b/w LAB of known values and image
            known_lab = row[0]
            image_lab = color[0][0]
            d = self.euclidean(known_lab, image_lab)
            # Update min. distante
            if d < minDist[0]:
                minDist = (d, i)

        # Name of color
        return self.colorNames[minDist[1]]

    def getDominantColors(self, img):
        # Reshape to RGB Pixels
        k_img = img.reshape((-1, 3))
        # Convert to float32
        k_img = np.float32(k_img)
        # KMeans setup
        flags = cv2.KMEANS_RANDOM_CENTERS
        num_colors = 3
        attempts = 10
        max_iterations = 50
        # How much of a difference between means before considered accurate enough
        epsilon = 0.0001
        # Either epsilon or iterations reached
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    max_iterations,
                    epsilon)
        # Apply KMeans
        ret, labels, centers = cv2.kmeans(k_img, num_colors, None, criteria,
                                          attempts, flags)
        return ret, labels, centers


class SegmentImageRos:
    def __init__(self, debug=False):
        # CHANGE THE IMAGE TOPIC HERE
        image_topic = '/camera/color/image_raw'
        self.img_sub = rospy.Subscriber(image_topic, Image, self._image_callback)

        self.bridge = CvBridge()
        self._cv_img = None
        self.color_segmenter = ColorSegmenter()

        self.debug = debug

        self.colors_found = []

    def _image_callback(self, ros_img):
        # Convert the ROS Image msg to OpenCV format (Numpy array)
        try:
            self._cv_img = self.bridge.imgmsg_to_cv2(ros_img, 'bgr8')
        except CvBridgeError as e:
            rospy.logwarn("CVBridgeError Exception!\n{}".format(e))

    def getColorsFound(self):
        return self.colors_found

    def start(self):
        # Iterate over all objects
        while True:
            # Get kmeans of colors
            ret, labels, centers = self.color_segmenter.getDominantColors(self._cv_img)
            """ Get dominant color """
            # Default params
            dominant_color = [[[0, 0, 0]]]
            labeled_color = 'black'
            # Iterate over all colors found
            for center in centers:
                # Conver to LaB
                lab = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_BGR2LAB)
                labeled_color = self.color_segmenter.getColorLabel(lab)
                # We don't want black...
                if not labeled_color == 'black':
                    dominant_color = [[center]]
                    self.colors_found.append(labeled_color)
                    break

            """ Debug Display """
            if self.debug:
                # Display side by side
                w, h = self._cv_img.shape[:2]
                rgb_color = np.zeros((w, h, 3), np.uint8)
                r = dominant_color[0][0][0]
                g = dominant_color[0][0][1]
                b = dominant_color[0][0][2]
                rgb_color[:] = (r, g, b)
                vis = np.concatenate((self._cv_img, rgb_color), axis=1)
                # Window
                cv2.imshow(labeled_color, vis)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    rospy.init_node('color_segmentation')
    t = SegmentImageRos(True)
    t.start()
    colors = t.getColorsFound()
    num_red = colors.count('red')
    num_green = colors.count('green')
    num_blue = colors.count('blue')
    print "Colors: {}".format(colors)
    print "Red: {}\nBlue: {}\nGreen: {}".format(num_red, num_blue, num_green)
