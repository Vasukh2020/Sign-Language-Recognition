import cv2
import numpy as np
import os
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/A")
    os.makedirs("data/train/B")
    os.makedirs("data/train/C")
    os.makedirs("data/train/D")
    os.makedirs("data/train/E")
    os.makedirs("data/train/F")
    os.makedirs("data/train/G")
    os.makedirs("data/train/H")
    os.makedirs("data/train/I")
    os.makedirs("data/train/J")
    os.makedirs("data/train/K")
    os.makedirs("data/train/L")
    os.makedirs("data/train/M")
    os.makedirs("data/train/N")
    os.makedirs("data/train/O")
    os.makedirs("data/train/P")
    os.makedirs("data/train/Q")
    os.makedirs("data/train/R")
    os.makedirs("data/train/S")
    os.makedirs("data/train/T")
    os.makedirs("data/train/U")
    os.makedirs("data/train/V")
    os.makedirs("data/train/W")
    os.makedirs("data/train/X")
    os.makedirs("data/train/Y")
    os.makedirs("data/train/Z")
    os.makedirs("data/test/A")
    os.makedirs("data/test/B")
    os.makedirs("data/test/C")
    os.makedirs("data/test/D")
    os.makedirs("data/test/E")
    os.makedirs("data/test/F")
    os.makedirs("data/test/G")
    os.makedirs("data/test/H")
    os.makedirs("data/test/I")
    os.makedirs("data/test/J")
    os.makedirs("data/test/K")
    os.makedirs("data/test/L")
    os.makedirs("data/test/M")
    os.makedirs("data/test/N")
    os.makedirs("data/test/O")
    os.makedirs("data/test/P")
    os.makedirs("data/test/Q")
    os.makedirs("data/test/R")
    os.makedirs("data/test/S")
    os.makedirs("data/test/T")
    os.makedirs("data/test/U")
    os.makedirs("data/test/V")
    os.makedirs("data/test/W")
    os.makedirs("data/test/X")
    os.makedirs("data/test/Y")
    os.makedirs("data/test/Z")
mode = 'train'
directory = 'data/'+mode+'/'
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1500)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    count = {'a': len(os.listdir(directory+"/A")),
             'b': len(os.listdir(directory+"/B")),
             'c': len(os.listdir(directory+"/C")),
             'd': len(os.listdir(directory+"/D")),
             'e': len(os.listdir(directory+"/E")),
             'f': len(os.listdir(directory+"/F")),
             'g': len(os.listdir(directory+"/G")),
             'h': len(os.listdir(directory+"/H")),
             'i': len(os.listdir(directory+"/I")),
             'j': len(os.listdir(directory+"/J")),
             'k': len(os.listdir(directory+"/K")),
             'l': len(os.listdir(directory+"/L")),
             'm': len(os.listdir(directory+"/M")),
             'n': len(os.listdir(directory+"/N")),
             'o': len(os.listdir(directory+"/O")),
             'p': len(os.listdir(directory+"/P")),
             'q': len(os.listdir(directory+"/Q")),
             'r': len(os.listdir(directory+"/R")),
             's': len(os.listdir(directory+"/S")),
             't': len(os.listdir(directory+"/T")),
             'u': len(os.listdir(directory+"/U")),
             'v': len(os.listdir(directory+"/V")),
             'w': len(os.listdir(directory+"/W")),
             'x': len(os.listdir(directory+"/X")),
             'y': len(os.listdir(directory+"/Y")),
             'z': len(os.listdir(directory+"/Z"))}

    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "A : "+str(count['a']), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "B : "+str(count['b']), (10, 140), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 1)
    cv2.putText(frame, "C : "+str(count['c']), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "D : "+str(count['d']), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "E : "+str(count['e']), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "F : "+str(count['f']), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "G : "+str(count['g']), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "H : "+str(count['h']), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "I : "+str(count['i']), (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "J : "+str(count['j']), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "K : "+str(count['k']), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "L : "+str(count['l']), (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "M : "+str(count['m']), (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "N : "+str(count['n']), (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "O : "+str(count['o']), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "P : "+str(count['p']), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "Q : "+str(count['q']), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "R : "+str(count['r']), (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "S : "+str(count['s']), (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "T : "+str(count['t']), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "U : "+str(count['u']), (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "V : "+str(count['v']), (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "W : "+str(count['w']), (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "X : "+str(count['x']), (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "Y : "+str(count['y']), (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(frame, "Z : "+str(count['z']), (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
   
    x1 = int(0.75*frame.shape[1])
    y1 = 10
    x2 =frame.shape[1]-10
    y2 = int(0.25*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (60, 60)) 
    cv2.imshow("Frame", frame)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 130, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: 
        break
    if interrupt & 0xFF == ord('A'):
        cv2.imwrite(directory+'A/'+str(count['a'])+'.jpg', roi)
    if interrupt & 0xFF == ord('B'):
        cv2.imwrite(directory+'B/'+str(count['b'])+'.jpg', roi)
    if interrupt & 0xFF == ord('C'):
        cv2.imwrite(directory+'C/'+str(count['c'])+'.jpg', roi)
    if interrupt & 0xFF == ord('D'):
        cv2.imwrite(directory+'D/'+str(count['d'])+'.jpg', roi)
    if interrupt & 0xFF == ord('E'):
        cv2.imwrite(directory+'E/'+str(count['e'])+'.jpg', roi)
    if interrupt & 0xFF == ord('F'):
        cv2.imwrite(directory+'F/'+str(count['f'])+'.jpg', roi)
    if interrupt & 0xFF == ord('G'):
        cv2.imwrite(directory+'G/'+str(count['g'])+'.jpg', roi)
    if interrupt & 0xFF == ord('H'):
        cv2.imwrite(directory+'H/'+str(count['h'])+'.jpg', roi)
    if interrupt & 0xFF == ord('I'):
        cv2.imwrite(directory+'I/'+str(count['i'])+'.jpg', roi)
    if interrupt & 0xFF == ord('J'):
        cv2.imwrite(directory+'J/'+str(count['j'])+'.jpg', roi)
    if interrupt & 0xFF == ord('K'):
        cv2.imwrite(directory+'K/'+str(count['k'])+'.jpg', roi)
    if interrupt & 0xFF == ord('L'):
        cv2.imwrite(directory+'L/'+str(count['l'])+'.jpg', roi)
    if interrupt & 0xFF == ord('M'):
        cv2.imwrite(directory+'M/'+str(count['m'])+'.jpg', roi)
    if interrupt & 0xFF == ord('N'):
        cv2.imwrite(directory+'N/'+str(count['n'])+'.jpg', roi)
    if interrupt & 0xFF == ord('O'):
        cv2.imwrite(directory+'O/'+str(count['o'])+'.jpg', roi)
    if interrupt & 0xFF == ord('P'):
        cv2.imwrite(directory+'P/'+str(count['p'])+'.jpg', roi)
    if interrupt & 0xFF == ord('Q'):
        cv2.imwrite(directory+'Q/'+str(count['q'])+'.jpg', roi)
    if interrupt & 0xFF == ord('R'):
        cv2.imwrite(directory+'R/'+str(count['r'])+'.jpg', roi)
    if interrupt & 0xFF == ord('S'):
        cv2.imwrite(directory+'S/'+str(count['s'])+'.jpg', roi)
    if interrupt & 0xFF == ord('T'):
        cv2.imwrite(directory+'T/'+str(count['t'])+'.jpg', roi)
    if interrupt & 0xFF == ord('U'):
        cv2.imwrite(directory+'U/'+str(count['u'])+'.jpg', roi)
    if interrupt & 0xFF == ord('V'):
        cv2.imwrite(directory+'V/'+str(count['v'])+'.jpg', roi)
    if interrupt & 0xFF == ord('W'):
        cv2.imwrite(directory+'W/'+str(count['w'])+'.jpg', roi)
    if interrupt & 0xFF == ord('X'):
        cv2.imwrite(directory+'X/'+str(count['x'])+'.jpg', roi)
    if interrupt & 0xFF == ord('Y'):  
        cv2.imwrite(directory+'Y/'+str(count['y'])+'.jpg', roi)
    if interrupt & 0xFF == ord('Z'):
        cv2.imwrite(directory+'Z/'+str(count['z'])+'.jpg', roi)
cap.release()
cv2.destroyAllWindows()