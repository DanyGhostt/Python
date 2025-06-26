import cv2

# Rutas 
orig = cv2.imread("C:/Users/juand/OneDrive/Desktop/PracticasTopicos/practica1-10/imagen_003.jpg")
template = cv2.imread("C:/Users/juand/OneDrive/Desktop/PracticasTopicos/practica1-10/template_003.jpg")

# Validar 
if orig is None or template is None:
    exit()

# Redimensionar 
scale_percent = 50  
width = int(orig.shape[1] * scale_percent / 100)
height = int(orig.shape[0] * scale_percent / 100)
orig = cv2.resize(orig, (width, height))
template = cv2.resize(template, (int(template.shape[1] * scale_percent / 100), int(template.shape[0] * scale_percent / 100)))

image = orig.copy()

# Escala de grises
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

methods = [
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED
]

for method in methods:
    res = cv2.matchTemplate(image_gray, template_gray, method=method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED:
        x1, y1 = min_loc
        x2 = x1 + template.shape[1]
        y2 = y1 + template.shape[0]
    else:
        x1, y1 = max_loc
        x2 = x1 + template.shape[1]
        y2 = y1 + template.shape[0]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Mostrar método aplicado 
    #method_name = str(method)
    #cv2.putText(image, method_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("image", image)
    cv2.imshow("Template", template)
    image = orig.copy()
    cv2.waitKey(0)

cv2.destroyAllWindows()
