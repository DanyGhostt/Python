import cv2
import numpy as np

def figColor(imagenHSV):
    rosaBajo = np.array([140, 50, 70], np.uint8)
    rosaAlto = np.array([170, 255, 255], np.uint8)
    maskRosa = cv2.inRange(imagenHSV, rosaBajo, rosaAlto)
    cntsRosa = cv2.findContours(maskRosa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(cntsRosa) > 0:
        color = 'Rosa'
    else:
        color = 'Desconocido'
    return color

def figName(contorno, width, height):
    epsilon = 0.01 * cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, epsilon, True)
    if len(approx) == 3:
        namefig = 'Triangulo'
    elif len(approx) == 4:
        aspect_ratio = float(width) / height
        namefig = 'Cuadrado' if 0.95 <= aspect_ratio <= 1.05 else 'Rectangulo'
    elif len(approx) == 5:
        namefig = 'Pentagono'
    elif len(approx) == 6:
        namefig = 'Hexagono'
    elif len(approx) > 10:
        namefig = 'Circulo'
    else:
        namefig = 'Otra'
    return namefig


##########################
# BLOQUE 1: Imagen estática
##########################
# imagen = cv2.imread("C:/Users/juand/OneDrive/Desktop/PracticasTopicos/practica 1-11/figurasDcolores.png")
# if imagen is None:
#     print("Error: No se pudo cargar la imagen.")
#     exit()
# gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(gray, 10, 150)
# canny = cv2.dilate(canny, None, iterations=1)
# canny = cv2.erode(canny, None, iterations=1)
# cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# imagenHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
# for c in cnts:
#     x, y, w, h = cv2.boundingRect(c)
#     imAux = np.zeros(imagen.shape[:2], dtype="uint8")
#     imAux = cv2.drawContours(imAux, [c], -1, 255, -1)
#     maskHSV = cv2.bitwise_and(imagenHSV, imagenHSV, mask=imAux)
#     color = figColor(maskHSV)
#     name = figName(c, w, h)
#     nameColor = f"{name} {color}"
#     cv2.putText(imagen, nameColor, (x, y - 5), 1, 0.8, (0, 255, 0), 1)
#     cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 2)
# cv2.imshow('Resultado Imagen', imagen)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##########################
# BLOQUE 2: Video grabado
##########################
# cap = cv2.VideoCapture("C:/Users/juand/OneDrive/Desktop/PracticasTopicos/practica 1-11/video_prueba.mp4")
# if not cap.isOpened():
#     print("Error: No se pudo abrir el video.")
#     exit()
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     canny = cv2.Canny(gray, 10, 150)
#     canny = cv2.dilate(canny, None, iterations=1)
#     canny = cv2.erode(canny, None, iterations=1)
#     cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     imagenHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     for c in cnts:
#         x, y, w, h = cv2.boundingRect(c)
#         imAux = np.zeros(frame.shape[:2], dtype="uint8")
#         imAux = cv2.drawContours(imAux, [c], -1, 255, -1)
#         maskHSV = cv2.bitwise_and(imagenHSV, imagenHSV, mask=imAux)
#         color = figColor(maskHSV)
#         name = figName(c, w, h)
#         nameColor = f"{name} {color}"
#         cv2.putText(frame, nameColor, (x, y - 5), 1, 0.8, (0, 255, 0), 1)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     cv2.imshow('Resultado Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


##########################
# BLOQUE 3: Video streaming en vivo (webcam)
##########################
cap = cv2.VideoCapture(0)
if not cap.isOpened():
     print("Error: No se pudo abrir la cámara.")
     exit()
while True:
     ret, frame = cap.read()
     if not ret:
         break
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     canny = cv2.Canny(gray, 10, 150)
     canny = cv2.dilate(canny, None, iterations=1)
     canny = cv2.erode(canny, None, iterations=1)
     cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     imagenHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     for c in cnts:
         x, y, w, h = cv2.boundingRect(c)
         imAux = np.zeros(frame.shape[:2], dtype="uint8")
         imAux = cv2.drawContours(imAux, [c], -1, 255, -1)
         maskHSV = cv2.bitwise_and(imagenHSV, imagenHSV, mask=imAux)
         color = figColor(maskHSV)
         name = figName(c, w, h)
         nameColor = f"{name} {color}"
         cv2.putText(frame, nameColor, (x, y - 5), 1, 0.8, (0, 255, 0), 1)
         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
     cv2.imshow('Resultado Streaming', frame)
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()
