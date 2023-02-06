#pls remember to install both opencv-python and numpy
from __future__ import print_function
import numpy
import cv2
import numpy as np

global pts_dst
#Create the parser(filter) and

im_src = cv2.imread('dress.png')
size = im_src.shape


#Grab the video from your LD video feed (Might have to change the number in the bracket if it doesn't work)
video = cv2.VideoCapture(0)

#Add in classifiers that identify human features from a database which we pull from OpenCV (The module is HOG and I think its still from Harr?)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#hog.setSVMDetector() method sets coefficients for the linear SVM classifier.

#Body detection magic happens down here
    #return each frame
ret, frame = video.read()

# Detect faces in the image (cOoL i KnOw)
print("here")

(humans, _) = hog.detectMultiScale(frame,
                                    winStride=(5, 5),
                                    padding=(3, 3),
                                    scale=1.21)

for (x, y, w, h) in humans:
    cv2.rectangle = (frame, (x, y),
                    (x + w, y + h),
                    (0, 0, 255), 2)
    print((x + w, y + h))
    pts_dst = (x + w, y + h)
    print(pts_dst)

print(pts_dst)


pts_src = np.array(
    [
        [0, 0],
        [size[1] - 1, 0],
        [size[1] - 1, size[0] - 1],
        [0, size[0] - 1]
    ], dtype=float
);

im_dst = frame
        # Show the video feed =
h, status = cv2.findHomography(pts_src, pts_dst)
temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16);
im_dst = im_dst + temp
cv2.imshow('Display', im_dst)

#if cv2.waitKey(1) == ord('q'):
    #break
# When everything done, release the capture
video.release()
cv2.destroyAllWindows()