import cv2 as cv
import cv2
import numpy as np
from collections import deque
import time
import math

import pigpio

pi = pigpio.pi() # Connect to local Pi.


#DEFINE CONSTANTS
horServoPIN = 18
vertServoPIN = 12
verticalScanningAngle = 1500
diagFOV = np.radians(40) #diagonal field of view

pulse_width_x = verticalScanningAngle  # starting position of the servos
pulse_width_y = 1100

diagFOV = np.radians(40) #diagonal field of view
horFOV = 2 * math.atan(math.tan(diagFOV/2) * math.cos(math.atan(480/640)))  # horizontal field of view
vertFOV = 2 * math.atan(math.tan(diagFOV/2) * math.sin(math.atan(480/640)))

lastDetectTime = -1 # variable which contains the time since last facial detection
scanDirection = "left" # means that it will start out scanning left
lastScanTime = -1 # variable that contains the last time a scan update was done

#CUSTOM CONVERSION FUNCTIONS
def get_angle_diff(centroid):
    """
    centroid: the x, y coordinate of the centroid of a bounding box
    return: (the x_angle_difference between centroid[0] the center of the image, 
        the y_angle difference between centroid[1] and the center of the image)

    """
    return np.degrees(centroid[0]/640*horFOV - horFOV/2), np.degrees(vertFOV/2 - centroid[1]/480*vertFOV)

def convertAngleToPulseWidth(angle):
    """
    angle: the angle in degrees which will be converted into its equivalent amount in pulse width
    """
    return angle/180 * 2000

# Load the model.
net = cv.dnn.readNet('face-detection-adas-0001.xml',
                     'face-detection-adas-0001.bin')
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
# Specify target device.
cap = cv2.VideoCapture(0)


pi.set_servo_pulsewidth(horServoPIN, pulse_width_x)
pi.set_servo_pulsewidth(vertServoPIN, pulse_width_y)

sentRequest = False #  are you awaiting a classification from ncs2?
detectionList = [] # list of the most recent detections (x, y)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not sentRequest: # if ncs2 resource avail, ask for classification net to run on last image
        blob = cv.dnn.blobFromImage(frame, size=(672,384), ddepth=cv.CV_8U)
        net.setInput(blob)
        out = net.forwardAsync()
        sentRequest = True
    
    if out.wait_for(0): # if the classification result is done
        out = out.get()
        detectionList = []
        i = 0
        detections = out.reshape(-1, 7)
        centroidList = [] #  keep a list of valid detection
        for detection in detections:
            #for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            if confidence > .7:  # only add bounding boxes of frames with a high confidence
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])
                centroid = ((xmax + xmin)/2, (ymax + ymin)/2)
                detectionList.append((xmin, ymin, xmax, ymax))
                centroidList.append(centroid)

        if len(centroidList) != 0:
            lastDetectTime = time.time()
            centroid = sorted(centroidList, key = lambda x: x[0])[0] #  always choose predictions on one side of camera

            angleDiff = get_angle_diff(centroid)
            
            # Horizontal Servo Calculations
            pulse_x_change = convertAngleToPulseWidth(angleDiff[0]) 
            if abs(pulse_x_change) < 20: # if its a mini update to posistion ignore it
                pulse_x_change = 0
            pulse_width_x -= pulse_x_change

            pulse_width_x = min(2500, int(pulse_width_x))  # 2500 is max deg rotation
            pulse_width_x = max(500, int(pulse_width_x))  #500 is min deg rotation

            # Vertical Servo Calculations
            pulse_y_change = convertAngleToPulseWidth(angleDiff[1]) 
            if abs(pulse_y_change) < 20:
                pulse_y_change = 0

            pulse_width_y -= pulse_y_change
            pulse_width_y = min(2500, int(pulse_width_y))  # 2500 is max deg rotation
            pulse_width_y = max(500, int(pulse_width_y))  #500 is min deg rotation

            if abs(pulse_x_change) < 100 and abs(pulse_y_change) < 100:
                print("You are on target")

            #Change Servo pos
            pi.set_servo_pulsewidth(horServoPIN, pulse_width_x)
            pi.set_servo_pulsewidth(vertServoPIN, pulse_width_y)

        sentRequest = False

    # Scanning Routine
    if time.time() - lastDetectTime > 1 and time.time() - lastScanTime > .1: # if it hasen't seen a person and scanned in a while
        pulse_width_y = verticalScanningAngle # scanning angle
        pi.set_servo_pulsewidth(vertServoPIN, pulse_width_y) #
        if scanDirection == "left":
            pulse_width_x -= 20
            lastScanTime = time.time()
        elif time.time():
            pulse_width_x += 20
            lastScanTime = time.time()

        if pulse_width_x <= 500:
            pulse_width_x = 500
            scanDirection = "right"
        if pulse_width_x >= 2500:
            pulse_width_x = 2500
            scanDirection = "left"
        pi.set_servo_pulsewidth(horServoPIN, pulse_width_x)
    for xmin, ymin, xmax, ymax in detectionList: # add bounding boxes to frame might be a small lag
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
        
    cv2.imshow('frame',frame) # frames display with almost no lag
    if cv2.waitKey(1) & 0xFF == ord('q'): # this line of code allows for frame refresh
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
