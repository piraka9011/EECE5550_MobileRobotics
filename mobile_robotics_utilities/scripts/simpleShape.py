import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
import time

# Start video stream
picam = VideoStream(usePiCamera=True).start()
time.sleep(1)

while True:
    # Get image
    # frame = cv2.imread('image085447.jpg')
    frame = picam.read()
    frame = imutils.resize(frame, width=400)
    # Grayscale, threshold, then find contours
    # Contours are basically edges of the shape, point where two lines intersect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours we found and
    for cont in contours:
        approx = cv2.approxPolyDP(cont, 0.01 * cv2.arcLength(cont, True), True)
        lenApprox = len(approx)
        if lenApprox == 5:
            # print "Pentagon"
            cv2.drawContours(frame, [cont], -1, 255, 2)
        elif lenApprox == 4:
            # print "Square"
            cv2.drawContours(frame, [cont], -1, (0, 255, 0), 2)
        elif lenApprox == 3:
            # print "Triangle"
            cv2.drawContours(frame, [cont], -1, (0, 0, 255), 2)
        elif lenApprox == 9:
            # print "Half-Circle"
            cv2.drawContours(frame, [cont], -1, (255, 255, 0), 2)
        elif lenApprox > 15:
            # print "Circle"
            cv2.drawContours(frame, [cont], -1, (0, 255, 255), 2)

    # Show the image
    ts = time.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam.stop()
