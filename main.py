# detect the text from camera using tesseract and opencv
import cv2
import os
import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()
# set the path of img
img_path = os.path.join(os.path.dirname(__file__), 'img')
# camera capture images
image_sample_path = os.path.join(img_path, 'sample.jpg')
img_test_path = os.path.join(img_path, 'test.png')


# read the image
img = cv2.imread(image_sample_path, 0)
img_test = cv2.imread(img_test_path, 0)

RESIZED_LENGTH = 500
dim = (RESIZED_LENGTH, RESIZED_LENGTH)

# read text
pipeline.recognizer([img_test])


# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(resized, None)
kp2, des2 = orb.detectAndCompute(img_test, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
match = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in match:
    if m.distance < 0.6 * n.distance:
        good.append([m])

print(len(good))

img3 = cv2.drawMatchesKnn(resized, kp1, img_test, kp2, good, None, flags=2)

cv2.imshow('img', img3)
cv2.waitKey(0)