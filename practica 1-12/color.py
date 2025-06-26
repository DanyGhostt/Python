import cv2
import numpy as np

cap = cv2.VideoCapture(0)

azulBajo = np.array([100,100,20], np.uint8)
azulAlto = np.array([125,255,255], np.uint8)

while True:

  ret, frame = cap.read()

  if ret==True:
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frameHSV, azulBajo, azulAlto)
    
    cv2.imshow('maskAzul', mask)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('*'):
      break
cap.release()
cv2.destroyAllWindows()