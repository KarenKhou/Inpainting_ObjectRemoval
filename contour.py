import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("photo.png")

mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

copyimage = image.copy()

cv2.drawContours(copyimage, contours, -1, (255, 0, 255), -1)

alpha = 0.4
image_with_fill = cv2.addWeighted(copyimage, alpha, image, 1 - alpha, 0)

cv2.drawContours(image_with_fill, contours, -1, (255, 0, 255), 2)

plt.imshow(cv2.cvtColor(image_with_fill, cv2.COLOR_BGR2RGB))
plt.title("Contour + Remplissage")
plt.axis("off")
plt.show()
