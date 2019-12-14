# Auto-Turret
Face tracking pan-tilt camera coda with Neural Compute Stick 2

## * Hardware Requirements
+ Raspberry pi
+ Neural Compute Stick 2
+ pan-tilt camera with servo attached

## * Software Requirements
+ python
+ OpenVino Toolkit
+ cv2
+ numpy

## How to Run
sudo pigpiod \n
python Auto-Turret.py

I will add more documentation for this project in the future. 
Also, there are several improvements to be made with this code, including angle calculations. Feedback is welcome.

The code for the face-detection model comes from https://github.com/movidius/ncappzoo/ 
