import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1500)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    x1 = int(0.75*frame.shape[1])
    y1 = 10
    x2 =frame.shape[1]-10
    y2 = int(0.25*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,0,0) ,1)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64)) 
    cv2.imshow("Frame", frame)
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, test_image = cv2.threshold(roi, 130, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    prediction = {'A': result[0][0], 
                  'B': result[0][1], 
                  'C': result[0][2],
                  'D': result[0][3],
                  'E': result[0][4],
                  'F': result[0][5],
                  'G': result[0][6],
                  'H': result[0][7],
                  'I': result[0][8],
                  'J': result[0][9],
                  'K': result[0][10],
                  'L': result[0][11],
                  'M': result[0][12],
                  'N': result[0][13],
                  'O': result[0][14],
                  'P': result[0][15],
                  'Q': result[0][16],
                  'R': result[0][17],
                  'S': result[0][18],
                  'T': result[0][19],
                  'U': result[0][20],
                  'V': result[0][21],
                  'W': result[0][22],
                  'X': result[0][23],
                  'Y': result[0][24],
                  'Z': result[0][25]}
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    if prediction == result[0][3]:
                cv2.rectangle(image, (20, 300), (270, 425), (0, 0, 0), 1)
                cv2.putText(image, "0 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,)
        
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)    
    cv2.imshow("Frame", frame)
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: 
        break
cap.release()
cv2.destroyAllWindows()
