#! /home/upendra/anaconda3/bin/python
# USAGE
# python recognize_faces_video.py --encodings encodings002.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
from datetime import datetime
import time
import os
from google.cloud import storage
from firebase import firebase
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings",default="encodings002.pickle",
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

#database 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/upendra/EDI6/Face-recognition/face-recognition-a4f2e-5eed8804d09b.json"
firebase = firebase.FirebaseApplication("https://face-recognition-a4f2e.firebaseio.com/",None)
client = storage.Client()

bucket = client.get_bucket('face-recognition-a4f2e.appspot.com')

imageBlob = bucket.blob("/")

# load the known faces and embeddings

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
#data is dictioary which contains knownNames and knownEncodings

#create a dictionary for storing time of people entering the room
timeFlag = {}
img=0
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
cap= cv2.VideoCapture(0)
writer = None
time.sleep(2.0)
prevName = "Unknown"
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	frame1=frame
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
		
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)


		if (name in timeFlag.keys() and time.time() > timeFlag[name] + 30) or not name in timeFlag.keys():
			#get timestamp of function			
			timeFlag[name] = time.time()
			today=str(datetime.now())
			today=today.split(' ')

			#get a snapshots of person in images and save for a moment

			img+=1
			filename=name+today[0]+'_'+today[1]+'.jpg'
			cv2.imwrite(filename,frame1)
			
			print("saved")
			#upload saved image to firebase and delete local image
			imagePath = "/home/upendra/EDI6/Face-recognition/"+str(filename)
			imageBlob = bucket.blob("persons/"+str(filename))
			imageBlob.upload_from_filename(imagePath)
			print(name, "entered in room ! at: ", today[1], today[0])
			os.remove(imagePath)
			#enter name and time in logbook and photo in visited people database
			entry={
				'Name': name ,
				'date': today[0] ,
				'time': today[1]		
				}
			#print(entry)
			result= firebase.post('Logbook',entry)
			print(result)
	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)
###############################################################
	# if the writer is not None, write the frame with recognized
	# faces t odisk
	if writer is not None:
		writer.write(frame)

	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()
