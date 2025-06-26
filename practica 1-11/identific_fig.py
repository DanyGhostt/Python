import cv2
import numpy as np
import os

def figColor(imagenHSV):
    # Rango de colores en HSV
    rojoBajo1 = np.array([0 , 100, 20], np.uint8)
    rojoAlto1 = np.array([10 , 255, 255], np.uint8)
    rojoBajo2 = np.array([175 , 100, 20], np.uint8)
    rojoAlto2 = np.array([180 , 255, 255], np.uint8)

    naranjaBajo = np.array([11, 110, 20], np.uint8)
    naranjaAlto = np.array([19 , 255, 255], np.uint8)

    amarilloBajo = np.array([20, 100, 20], np.uint8)
    amarilloAlto = np.array([32, 255, 255], np.uint8)

    verdeBajo = np.array([36, 100, 20], np.uint8)
    verdeAlto = np.array([70, 255, 255], np.uint8)

    violetaBajo = np.array([130, 100, 20], np.uint8)
    violetaAlto = np.array([170, 255, 255], np.uint8)

    rosaBajo = np.array([140, 50, 70], np.uint8)
    rosaAlto = np.array([170, 255, 255], np.uint8)

    # Máscaras por color
    maskRojo1 = cv2.inRange(imagenHSV, rojoBajo1, rojoAlto1)
    maskRojo2 = cv2.inRange(imagenHSV, rojoBajo2, rojoAlto2)
    maskRojo = cv2.add(maskRojo1, maskRojo2)

    maskNaranja = cv2.inRange(imagenHSV, naranjaBajo, naranjaAlto)
    maskAmarillo = cv2.inRange(imagenHSV, amarilloBajo, amarilloAlto)
    maskVerde = cv2.inRange(imagenHSV, verdeBajo, verdeAlto)
    maskVioleta = cv2.inRange(imagenHSV, violetaBajo, violetaAlto)
    maskRosa = cv2.inRange(imagenHSV, rosaBajo, rosaAlto)

    # Contornos por color
    cntsRojo = cv2.findContours(maskRojo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntsNaranja = cv2.findContours(maskNaranja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntsAmarillo = cv2.findContours(maskAmarillo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntsVerde = cv2.findContours(maskVerde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntsVioleta = cv2.findContours(maskVioleta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntsRosa = cv2.findContours(maskRosa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    color = "Desconocido"
    if len(cntsRojo) > 0: color = 'Rojo'
    elif len(cntsNaranja) > 0: color = 'Naranja'
    elif len(cntsAmarillo) > 0: color = 'Amarillo'
    elif len(cntsVerde) > 0: color = 'Verde'
    elif len(cntsVioleta) > 0: color = 'Violeta'
    elif len(cntsRosa) > 0: color = 'Rosa'

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

# Cargar imagen (usa tu ruta si es necesario)
imagen = cv2.imread("C:/Users/juand/OneDrive/Desktop/PracticasTopicos/practica 1-11/figurasDcolores.png")

# Validar carga
if imagen is None:
    print(" Error: No se pudo cargar la imagen. Verifica la ruta o el nombre del archivo.")
    exit()

gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 10, 150)
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)

cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imagenHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    imAux = np.zeros(imagen.shape[:2], dtype="uint8")
    imAux = cv2.drawContours(imAux, [c], -1, 255, -1)
    maskHSV = cv2.bitwise_and(imagenHSV, imagenHSV, mask=imAux)
    color = figColor(maskHSV)
    name = figName(c, w, h)
    nameColor = f"{name} {color}"
    cv2.putText(imagen, nameColor, (x, y - 5), 1, 0.8, (0, 255, 0), 1)
    cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Resultado', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
