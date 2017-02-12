import cv2, os
import numpy as np
import matplotlib.image as mpimg

image_dir = 'data'

image_height, image_width, image_channels = 66, 200, 3
input_shape = (image_height, image_width, image_channels)


def load_image(file):
    """
    Load RGB images from a file
    """
    path = os.path.join(image_dir, file.strip())
    image = mpimg.imread(path)
    return image


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[30:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(left), steering_angle + 0.2
    elif choice == 1:
        return load_image(right), steering_angle - 0.2
    return load_image(center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def augument(center, left, right, steering_angle):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, 100, 10)
    return image, steering_angle


def batch_generator(image_paths, steering_angles, batch_size):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, image_height, image_width, image_channels])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            image, steering_angle = augument(center, left, right, steering_angle)
            # Most of the training images has zero steering angle.
            # Here, we are making sure we have more non-zero steering angles.
            if abs(steering_angle) < 0.01 and np.random.rand() < 0.7:
                while abs(steering_angle) < 0.01:
                    image, steering_angle = augument(center, left, right, steering_angle)
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

