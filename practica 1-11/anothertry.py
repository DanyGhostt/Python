import cv2
import numpy as np

def figColor(imagenHSV):
    colores = {
        'Rojo': [([0, 100, 20], [10, 255, 255]), ([175, 100, 20], [180, 255, 255])],
        'Naranja': [([11, 110, 20], [19, 255, 255])],
        'Amarillo': [([20, 100, 20], [32, 255, 255])],
        'Verde': [([36, 100, 20], [70, 255, 255])],
        'Violeta': [([130, 100, 20], [170, 255, 255])],
        'Rosa': [([140, 50, 70], [170, 255, 255])]
    }
    
    min_area = 500  # área mínima para considerar color
    
    for color, rangos in colores.items():
        mask_total = None
        for bajo, alto in rangos:
            bajo_np = np.array(bajo, dtype=np.uint8)
            alto_np = np.array(alto, dtype=np.uint8)
            mask = cv2.inRange(imagenHSV, bajo_np, alto_np)
            if mask_total is None:
                mask_total = mask
            else:
                mask_total = cv2.bitwise_or(mask_total, mask)
        cnts = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area:
                return color
    return 'Desconocido'

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
# BLOQUE 3: Video streaming en vivo (webcam)
##########################

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

min_area = 500  # área mínima para filtrar ruido

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gray, 10, 150)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)

    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagenHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

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
