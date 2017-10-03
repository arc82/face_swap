# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import numpy as np

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
        # frame = cv2.imread("abba.png")

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces)))

        # output = gray.copy()

        if len(faces) > 1:
            x0 = faces[0, 0]
            y0 = faces[0, 1]
            w0 = faces[0, 2]
            h0 = faces[0, 3]
            x1 = faces[1, 0]
            y1 = faces[1, 1]
            w1 = faces[1, 2]
            h1 = faces[1, 3]
            face0 = np.copy(frame[y0:y0+h0, x0:x0+h0])
            face1 = np.copy(frame[y1:y1+h1, x1:x1+w1])

            face0_resized = cv2.resize(face0, (w1, h1), interpolation = cv2.INTER_AREA)
            face1_resized = cv2.resize(face1, (w0, h0), interpolation = cv2.INTER_AREA)
            
            frame[y0:y0+h0, x0:x0+w0] = face1_resized
            frame[y1:y1+h1, x1:x1+w1] = face0_resized

	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
