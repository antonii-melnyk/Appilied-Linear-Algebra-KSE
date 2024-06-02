import matplotlib.pyplot as plt
import numpy as np

object1 = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
object2 = np.array([[0, 0], [2, 0.5], [0.8, 2], [1, 0.8], [0, 1.6], [-1, 0.8], [-0.8, 2], [-2, 0.5], [0, 0]])

plt.figure()
plt.plot(object1[:, 0], object1[:, 1], label='Object 1')
plt.plot(object2[:, 0], object2[:, 1], label='Object 2')
plt.legend()
plt.title('Initial Objects')
plt.show()

def rotate(points, angle):
    rad = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return np.dot(points, rotation_matrix.T)

def scale(points, factor):
    scale_matrix = np.array([[factor, 0], [0, factor]])
    return np.dot(points, scale_matrix.T)

def reflect(points, axis):
    if axis == 'x':
        reflect_matrix = np.array([[1, 0], [0, -1]])
    elif axis == 'y':
        reflect_matrix = np.array([[-1, 0], [0, 1]])
    else:
        raise ValueError("Axis must be 'x' or 'y'")
    return np.dot(points, reflect_matrix.T)

def shear(points, shear_factor, axis):
    if axis == 'x':
        shear_matrix = np.array([[1, shear_factor], [0, 1]])
    elif axis == 'y':
        shear_matrix = np.array([[1, 0], [shear_factor, 1]])
    else:
        raise ValueError("Axis must be 'x' or 'y'")
    return np.dot(points, shear_matrix.T)

def custom_transform(points, matrix):
    return np.dot(points, matrix.T)

transformed_object1 = rotate(object1, 45)
plt.figure()
plt.plot(transformed_object1[:, 0], transformed_object1[:, 1], label='Rotated Object 1')
plt.legend()
plt.title('Rotated Object 1')
plt.show()

custom_matrix = np.array([[2, 0.5], [-0.5, 1]])
transformed_object1 = custom_transform(object1, custom_matrix)

plt.figure()
plt.plot(transformed_object1[:, 0], transformed_object1[:, 1], label='Transformed Object 1 with Custom Matrix')
plt.legend()
plt.title('Custom Transformation')
plt.show()

scaled_object1 = scale(object1, 0.5)
scaled_object2 = scale(object2, 0.5)

plt.figure()
plt.plot(scaled_object1[:, 0], scaled_object1[:, 1], label='Scaled Object 1')
plt.plot(scaled_object2[:, 0], scaled_object2[:, 1], label='Scaled Object 2')
plt.legend()
plt.title('Scaled Objects')
plt.show()

reflected_object1_x = reflect(object1, 'x')
reflected_object1_y = reflect(object1, 'y')

plt.figure()
plt.plot(reflected_object1_x[:, 0], reflected_object1_x[:, 1], label='Reflected Object 1 (x-axis)')
plt.plot(reflected_object1_y[:, 0], reflected_object1_y[:, 1], label='Reflected Object 1 (y-axis)')
plt.legend()
plt.title('Reflected Objects')
plt.show()

object3d = np.array([[0, 0, 0], [1, 0.2, 0.5], [0.4, 1, 1], [0.5, 0.4, 1.5], [0, 0.8, 2], [-0.5, 0.4, 1.5], [-0.4, 1, 1], [-1, 0.2, 0.5], [0, 0, 0]])

def rotate3d(points, angle, axis):
    rad = np.deg2rad(angle)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")
    return np.dot(points, rotation_matrix.T)

def scale3d(points, factor):
    scale_matrix = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, factor]])
    return np.dot(points, scale_matrix.T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(object3d[:, 0], object3d[:, 1], object3d[:, 2], label='3D Object')
ax.legend()
plt.title('3D Object')
plt.show()

transformed_object3d = rotate3d(object3d, 45, 'z')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(transformed_object3d[:, 0], transformed_object3d[:, 1], transformed_object3d[:, 2], label='Rotated 3D Object')
ax.legend()
plt.title('Rotated 3D Object')
plt.show()

scaled_object3d = scale3d(object3d, 0.5)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(scaled_object3d[:, 0], scaled_object3d[:, 1], scaled_object3d[:, 2], label='Scaled 3D Object')
ax.legend()
plt.title('Scaled 3D Object')
plt.show()