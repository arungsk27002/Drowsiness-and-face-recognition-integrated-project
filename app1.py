import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils
import requests
import pickle
import time
from imutils.video import FPS
from imutils.video import VideoStream

blynk_api_url = "https://blr1.blynk.cloud/external/api/update?token=CThsFrrrZV0wwkPnt_NHhvTOYmxP-NVA&v1="
face_api_url = "https://blr1.blynk.cloud/external/api/update?token=CThsFrrrZV0wwkPnt_NHhvTOYmxP-NVA&v2="
delay_seconds = 1
start_time = time.time()
face_frame=0
previous_millis=time.time()
interval=5

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)


def blynk_update(blynk_value,blynk_api_url=blynk_api_url):
    full_url = f"{blynk_api_url}{blynk_value}"
    response = requests.get(full_url)
    if response.status_code == 200:
        print(f"Value updated successfully to {blynk_value}")
    else:
        print(f"Failed to update value. Status code: {response.status_code}")
        print(response.text)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up/(2.0*down)

    # Checking if it is blinked
    if (ratio > 0.25):
        return 2
    elif (ratio > 0.21 and ratio <= 0.25):
        return 1
    else:
        return 0

def face_recog(frames,current_time):
	global previous_millis
    # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frames = imutils.resize(frames, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frames, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			if(name=="dixon"):
				name="Unknown"

			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			print("detected person =", name)
			if(current_time-previous_millis>=interval):
				previous_millis=current_time
				if(name=="aswin"):
					blynk_update(100,face_api_url)
				else:
					blynk_update(99,face_api_url)
		else:
			if(current_time-previous_millis>=interval):
				previous_millis=current_time
				blynk_update(99,face_api_url)
                    
            

	    # update the FPS counte
		fps.update()    
	# show the output frame
	cv2.imshow("Face recognition", frame)

while True:
    current_time = time.time()
    frame = vs.read()
    face_recog(frame,current_time)
    frame = imutils.resize(frame, width=600)  # Resize frame if necessary
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector_dlib(gray)
    # detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36], landmarks[37],landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Now judge what to do for the eye blinks
        if (left_blink == 0 or right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep > 6):
                status = "Drowsy !"
                color = (255, 0, 0)
                if current_time - start_time >= delay_seconds:
                    start_time = current_time
                    blynk_update(50)

        elif (left_blink == 1 or right_blink == 1):
            sleep = 0
            active = 0
            drowsy += 1
            if (drowsy > 6):
                status = "Drowsy !"
                color = (255, 0, 0)
                if current_time - start_time >= delay_seconds:
                    start_time = current_time
                    blynk_update(50)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 6):
                status = "Active :)"
                color = (0, 255, 0)
                if current_time - start_time >= delay_seconds:
                    start_time = current_time
                    blynk_update(0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    if cv2.waitKey(1) == ord('q'):
        break

fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()
