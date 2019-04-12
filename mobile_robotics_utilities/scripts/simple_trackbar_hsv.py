#!/usr/bin/env python
"""Simple example for using trackbars. Segments image color based on HSV."""
__author__ = "Anas Abou Allaban"
__maintainer__ = "Anas Abou Allaban"
__email__ = "anas@abouallaban.info"

import cv2
import numpy as np
import time

trackbarWindowName = 'Trackbars'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_NUM_OBJECTS = 50
MIN_OBJECT_AREA = 20*20
MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5
H_MIN = 0; S_MIN = 0; V_MIN = 0
H_MAX = 179; S_MAX = 255; V_MAX = 255
BLUE_HSV_MIN = (100, 0, 0)
BLUE_HSV_MAX = (179, 255, 45)
RED_HSV_MIN = (0, 0, 71)
RED_HSV_MAX = (179, 24, 141)
GREEN_HSV_MIN = (19, 0, 0)
GREEN_HSV_MAX = (46, 47, 24)


def onTrackbar(x):
    pass


def createTrackbars():
    cv2.namedWindow(trackbarWindowName)
    cv2.createTrackbar('H_MIN', trackbarWindowName, 0, 255, onTrackbar)
    cv2.createTrackbar('H_MAX', trackbarWindowName, 179, 179, onTrackbar)
    cv2.createTrackbar('S_MIN', trackbarWindowName, 0, 255, onTrackbar)
    cv2.createTrackbar('S_MAX', trackbarWindowName, 255, 255, onTrackbar)
    cv2.createTrackbar('V_MIN', trackbarWindowName, 0, 255, onTrackbar)
    cv2.createTrackbar('V_MAX', trackbarWindowName, 255, 255, onTrackbar)


def morphOps(thresh):
    erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, erodeElement, iterations=2)
    thresh = cv2.dilate(thresh, dilateElement, iterations=2)
    return thresh


def contourApproximation(threshold, cameraFeed):
    res, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(cameraFeed, [hull], 0, (0, 0, 255), 2)
        cv2.namedWindow('Approx')
        cv2.imshow('Approx', cameraFeed)
        cv2.moveWindow('Approx', 550, 550)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def findSquares(cameraFeed):
    threshold = cv2.inRange(cameraFeed, BLUE_HSV_MIN, BLUE_HSV_MAX)
    res, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        if len(cnt) == 3 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cv2.drawContours(cameraFeed, [cnt], -1, (0, 255, 0), 3)

        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in xrange(4)])
            if max_cos < 0.1:
                cv2.drawContours(cameraFeed, [cnt], -1, (0, 255, 0), 3)
        if len(cnt) and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cv2.drawContours(cameraFeed, [cnt], -1, (0, 255, 0), 3)
        cv2.namedWindow('Approx')
        cv2.imshow('Approx', cameraFeed)
        cv2.moveWindow('Approx', 550, 550)


def detect(contour):
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        # Compute aspect ratio to determine if square or rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        print "Square contour area: " + str(cv2.contourArea(contour))
    elif len(approx) == 5:
        shape = "pentagon"
    else:
        shape = "circle"
    return shape


def blueShapeDetect(cameraFeed):
    threshold = cv2.inRange(cameraFeed, BLUE_HSV_MIN, BLUE_HSV_MAX)
    res, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] and cv2.contourArea(cnt) > 30:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            shape = detect(cnt)
            cv2.drawContours(cameraFeed, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(cameraFeed, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.namedWindow('Blue')
            cv2.imshow('Blue', cameraFeed)
            cv2.moveWindow('Blue', 550, 550)


def redShapeDetect(cameraFeed):
    threshold = cv2.inRange(cameraFeed, RED_HSV_MIN, RED_HSV_MAX)
    res, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] and cv2.contourArea(cnt) > 30:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            shape = detect(cnt)
            cv2.drawContours(cameraFeed, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(cameraFeed, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.namedWindow('Red')
            cv2.imshow('Red', cameraFeed)
            cv2.moveWindow('Red', 550, 550)


def greenShapeDetect(cameraFeed):
    threshold = cv2.inRange(cameraFeed, GREEN_HSV_MIN, GREEN_HSV_MAX)
    res, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] and cv2.contourArea(cnt) > 30:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            shape = detect(cnt)
            cv2.drawContours(cameraFeed, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(cameraFeed, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
            cv2.namedWindow('Red')
            cv2.imshow('Red', cameraFeed)
            cv2.moveWindow('Red', 550, 550)


def trackFilteredObject(x, y, threshold, cameraFeed):
    # Find the contours in the image
    res, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
    objectFound = False
    refArea = 0
    if len(hierarchy) > 0:
        numObjects = len(hierarchy)
        if numObjects < MAX_NUM_OBJECTS:
            # Loop through all potential objects and calculate moments to find centroid
            for h in range(0, numObjects + 1):
                M = cv2.moments(contours[h])
                area = M['m00']
                if area > MIN_OBJECT_AREA and area < MAX_OBJECT_AREA and area > refArea:
                    x = M['10'] / area
                    y = M['01'] / area
                    objectFound = True
                    refArea = area
                else:
                    objectFound = False
            # Trace object if found
            if objectFound:
                cv2.putText(cameraFeed, 'Tracking Object', (0, 50), 2, 3, (0, 255, 0), 2)
                drawObject(x, y, cameraFeed)
                return x, y
            else:
                cv2.putText(cameraFeed, 'Too much noise, adjust filter', (0, 50), 1, 2, (0, 0, 255), 2)
                return x, y


def main():
    trackObjects = False
    useMorphOps = False
    x = 0
    y = 0
    createTrackbars()
    picam = cv2.VideoCapture(0)
    time.sleep(1)
    while True:
        ret, frame = picam.read()
        H_MIN = cv2.getTrackbarPos('H_MIN', trackbarWindowName)
        H_MAX = cv2.getTrackbarPos('H_MAX', trackbarWindowName)
        S_MIN = cv2.getTrackbarPos('S_MIN', trackbarWindowName)
        S_MAX = cv2.getTrackbarPos('S_MAX', trackbarWindowName)
        V_MIN = cv2.getTrackbarPos('V_MIN', trackbarWindowName)
        V_MAX = cv2.getTrackbarPos('V_MAX', trackbarWindowName)
        threshold = cv2.inRange(frame, (H_MIN, S_MIN, V_MIN), (H_MAX, S_MAX, V_MAX))
        mask = cv2.bitwise_and(frame, frame, mask=threshold)
        if useMorphOps:
            threshold = morphOps(threshold)
        if trackObjects:
            x, y = trackFilteredObject(x, y, threshold, frame)

        greenShapeDetect(frame)
        redShapeDetect(frame)
        cv2.namedWindow('Original Image')
        cv2.imshow('Original Image', frame)
        cv2.moveWindow('Original Image', 0, 0)
        cv2.namedWindow('Threshold')
        cv2.imshow('Threshold', threshold)
        cv2.moveWindow('Threshold', 0, 550)
        cv2.namedWindow('Mask')
        cv2.imshow('Mask', mask)
        cv2.moveWindow('Mask', 550, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    picam.stop()


if __name__ == '__main__':
    main()
