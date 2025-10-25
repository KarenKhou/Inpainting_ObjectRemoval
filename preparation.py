import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("photo2.png")
clone = image.copy()
mask = np.zeros(image.shape[:2], dtype=np.uint8)
drawing = False
prev_point = None
brush_size = 20

def draw_line(event, x, y, flags, param):
    global drawing, prev_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(clone, prev_point, (x, y), (0, 0, 255), brush_size)
        cv2.line(mask, prev_point, (x, y), 255, brush_size)
        prev_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(clone, prev_point, (x, y), (0, 0, 255), brush_size)
        cv2.line(mask, prev_point, (x, y), 255, brush_size)
        prev_point = None

cv2.namedWindow("Dessine le contour (s = sauver, r = reset, esc = quitter)")
cv2.setMouseCallback("Dessine le contour (s = sauver, r = reset, esc = quitter)", draw_line)

while True:
    cv2.imshow("Dessine le contour (s = sauver, r = reset, esc = quitter)", clone)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        clone = image.copy()
        mask[:] = 0
    elif key == ord('s'):
        break
    elif key == 27:
        exit()

cv2.destroyAllWindows()


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filled_mask = np.zeros_like(mask)
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

b, g, r = cv2.split(image)
alpha = np.ones_like(mask, dtype=np.uint8) * 255
alpha[filled_mask == 255] = 0


image_rgba = cv2.merge((b, g, r, alpha))


cv2.imwrite("mask.png", mask)
cv2.imwrite("image_incomplete.png", image_rgba)


plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(image_rgba, cv2.COLOR_BGRA2RGBA))
plt.title("Zone intérieure effacée (transparente)")
plt.axis("off")
plt.show()

print("✅ Masque sauvegardé sous mask.png")
print("✅ Image incomplète sauvegardée sous image_incomplete.png")
