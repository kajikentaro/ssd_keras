import cv2
import time
from keras.preprocessing import image
import numpy as np
cap = cv2.VideoCapture(0)
while True:
	time.sleep(1)

	ret, frame = cap.read()
	frame = cv2.resize(frame, (300, 300))
	frame = image.array_to_img(frame)

	cv2.waitKey(1)



cv2.destroyAllWindows()
cap.release()