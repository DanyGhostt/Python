import cv2
import numpy as np

# ----------- OPCIÓN 1: Imagen local --------------
# imagen = cv2.imread('C:/Users/juand/OneDrive/Desktop/PracticasTopicos/practica 1-13/figContorno.png')
# if imagen is None:
#     exit()
# gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
# _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
# contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(imagen, contornos, -1, (0, 255, 0), 3)
# cv2.imshow('imagen', imagen)
# cv2.imshow('th', th)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# ----------- OPCIÓN 2: STREAMING DESDE IP WEBCAM --------------
url = 'http://192.168.100.4:8080/video'  # ← CAMBIA esto a la IP de tu celular
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo acceder al streaming.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja los contornos en la imagen de streaming
    cv2.drawContours(frame, contornos, -1, (0, 255, 0), 2)

    cv2.imshow('Streaming con contornos', frame)
    cv2.imshow('Umbral', th)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
