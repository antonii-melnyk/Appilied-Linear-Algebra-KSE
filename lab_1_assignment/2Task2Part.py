import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Logo.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

h, w = image.shape[:2]
center = (w / 2, h / 2)
rotate_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, rotate_matrix, (w, h))

scaled_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

shear_matrix = np.float32([[1, 0.5, 0], [0, 1, 0]])
sheared_image = cv2.warpAffine(image, shear_matrix, (w + int(0.5 * h), h))

plt.figure()
plt.imshow(rotated_image)
plt.title('Rotated Image')
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(scaled_image)
plt.title('Scaled Image')
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(sheared_image)
plt.title('Sheared Image')
plt.axis('off')
plt.show()