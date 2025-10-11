import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Charger l'image originale et le masque du contour ---
image = cv2.imread("jaunebleu.png")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# --- 2. Remplir l'intérieur du contour blanc ---
# Trouver les contours du trait
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer un masque vide, même taille
filled_mask = np.zeros_like(mask)

# Remplir l'intérieur du contour en blanc
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

# --- 3. Créer un canal alpha basé sur ce masque rempli ---
b, g, r = cv2.split(image)
alpha = np.ones_like(mask, dtype=np.uint8) * 255
alpha[filled_mask == 255] = 0  # zone à supprimer devient transparente

# --- 4. Fusionner et sauvegarder ---
image_rgba = cv2.merge((b, g, r, alpha))
cv2.imwrite("image_incomplete.png", image_rgba)

# --- 5. Visualiser ---
plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(image_rgba, cv2.COLOR_BGRA2RGBA))
plt.title("Zone intérieure remplie et effacée (transparente)")
plt.axis("off")
plt.show()
