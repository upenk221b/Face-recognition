import cv2
import os
import time
from imutils.video import VideoStream
vs = VideoStream(src=0).start()
i=0
while True :
	frame = vs.read()
	cv2.imshow("Frame", frame)
	time.sleep(3)
	key = cv2.waitKey(1) & 0xFF
	cv2.imwrite('kang'+str(i)+'.jpg',frame)
	i+=1
	print("saved  ",time.time())



