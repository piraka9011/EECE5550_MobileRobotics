import cv2
import numpy as np
import yaml
import time

class ShapeColor:
    
    def __init__(self):
        self.trackbarWindowName = 'Trackbars'
        self.windowName = 'main'
        self.color2detect = 'blue'
        self.shape2detect = 'square'
        self.H_MIN = 0
        self.S_MIN = 0
        self.V_MIN = 0
        self.H_MAX = 179
        self.S_MAX = 255
        self.V_MAX = 255
        # Create main window
        cv2.namedWindow(self.windowName)
        # Open yaml file with params
        with open('ShapeColorParams', 'r') as stream:
            params = yaml.load(stream)
        self.color = params['color']
        self.shape = params['shape']
        self._MIN_CONTOUR_AREA = params['min_contour_area']
        self.boundPercentage = params['bound_percentage']

    def _recOrSq(self, approx):
        """Computes the aspect ratio to determine if shape is a square or rectangle"""
        (x, y, w, h) = cv2.boundingRect(approx)
        aspectRatio = w / float(h)
        return 'square' if 0.95 <= aspectRatio <= 1.05 else 'rectangle'

    def detect(self, color, shape, frame):
        """Detects the shape and color in the f-n args"""
        objectFound = False
        # Threshold image to get only color we want
        threshold = cv2.inRange(frame,
                                self.color[color]['min'], self.color[color]['max'])
        # Find all contours (edges)
        _, contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Calculate moments to get the centroids
            M = cv2.moments(cnt)
            # Make sure we don't divide by zero and the contour is a reasonable size
            if M['m00'] and cv2.contourArea(cnt) > self._MIN_CONTOUR_AREA:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                # Get the approximate shape of the detected feature
                epsilon = self.boundPercentage * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                numEdges = len(approx)
                # If we have 4 edges, check if its a rectangle or square. Otherwise process normally
                if numEdges == 4:
                    objectFound = True if self._recOrSq(approx) == self.shape[shape] else False
                elif numEdges == self.shape[shape]:
                    objectFound = True
                # Print shape if object was found
                if objectFound:
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                    cv2.putText(frame, self.shape[shape], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                    cv2.imshow(self.windowName, frame)
                else:
                    cv2.imshow(self.windowName, frame)

    def start(self):
        """Startup camera and detect a colored shape"""
        camera = cv2.VideoCapture(0)
        time.sleep(1)
        while True:
            # Get the next frame and detect
            _, frame = camera.read()
            self.detect(self.color[self.color2detect], self.shape[self.shape2detect], frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        camera.stop()
