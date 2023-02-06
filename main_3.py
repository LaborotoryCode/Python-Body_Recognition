#pls remember to install both opencv-python and numpy
from __future__ import print_function
import numpy as np
import cv2
import cv2.aruco as aruco

#Create the parser(filter) and

#Grab the video from your LD video feed (Might have to change the number in the bracket if it doesn't work)
video = cv2.VideoCapture(0)

#Add in classifiers that identify human features from a database which we pull from OpenCV (The module is HOG and I think its still from Harr?)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Body detection magic happens down here
while True:
    #return each frame
    ret, frame = video.read()
    #make the video feed gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #find aruco parameters (outline of person)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    arucoParameters =  aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
    if np.all(ids != None):
        display = aruco.drawDetectedMarkers(frame, corners)
        x1 = (corners[0][0][0][0], corners[0][0][0][1])
        x2 = (corners[0][0][1][0], corners[0][0][1][1])
        x3 = (corners[0][0][2][0], corners[0][0][2][1])
        x4 = (corners[0][0][3][0], corners[0][0][3][1])

    # Detect faces in the image
    bodies_detected = hog.detectMultiScale(gray_frame,
                                           winStride=(5, 5),
                                           padding=(3.3),
                                           scale=1.21)

    # Draw a rectangle around the body
    for (column, row, width, height) in bodies_detected:
        cv2.rectangle(frame, (column, row), (column + width, row + height), (0, 255, 0), 2)

    # Show the video feed (WHY WON'T IT BE GRAY)
    cv2.imshow('gray_frame', frame)


    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
video.release()
cv2.destroyAllWindows()