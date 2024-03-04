import cvzone
import pickle
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np


cap = cv2.VideoCapture(1)

FMD = FaceMeshDetector()



with open('model.pkl','rb') as f:
   Behaviour_model = pickle.load(f)


# taking video frame by frame
while cap.isOpened():
    rt,frame = cap.read()
    frame = cv2.resize(frame,(720,480))

    real_frame = frame.copy()

    img , faces = FMD.findFaceMesh(frame)
    cvzone.putTextRect(frame, ('Mood'), (10, 80))
    if faces:
        face = faces[0]
        face_data = list(np.array(face).flatten())
       

        try:
            # feeding newpoints to model for prediction
            result = Behaviour_model.predict([face_data])
            cvzone.putTextRect(frame, str(result[0]), (250, 80))
            print(result)

            # resultproba = Behaviour_model.predict_proba([face_data])
            # print(resultproba)
            

        except Exception as e:
            pass
    all_frames = cvzone.stackImages([real_frame,frame],2,0.70)
    cv2.imshow('frame',all_frames)
    cv2.waitKey(1)