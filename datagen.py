from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import csv

videopath = 'happy.mp4'
cap = cv2.VideoCapture(1)

FMD = FaceMeshDetector()
class_name = 'sad'

# Columns = ['Class']
# for val in range(1,468+1):
#     Columns += ['x{}'.format(val),'y{}'.format(val)]

# print(Columns)


# with open('data.csv','w',newline='') as f:
#     csv_writer = csv.writer(f,delimiter = ',')
#     csv_writer.writerow(Columns)


while cap.isOpened():
    rt,frame = cap.read()
    frame = cv2.resize(frame,(720,480))
    img , faces = FMD.findFaceMesh(frame)

    if faces:
        face = faces[0]
        face_data = list(np.array(face).flatten())
        face_data.insert(0,class_name)

        # print(face_data)
        # print(len(face_data))
        # print(face_data)

        with open('data.csv','a',newline='') as f:
            csv_writer = csv.writer(f,delimiter = ',')
            csv_writer.writerow(face_data)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyWindow()