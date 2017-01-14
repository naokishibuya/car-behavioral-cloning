import cv2, os
import numpy as np
import matplotlib.image as mpimg

image_dir = 'data'

image_height, image_width, image_channels = 66, 200, 3
input_shape = (image_height, image_width, image_channels)


def load_image(file):
    path = os.path.join(image_dir, file.strip())
    image = mpimg.imread(path)
    return image


def crop(image):
    return image[30:-25, :, :] # remove the sky and the car front


def resize(image):
    return cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)


def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(center, left, right, steering):
    choice = np.random.choice(3)
    if choice==0:
        return load_image(left), steering + 0.2
    elif choice==1:
        return load_image(right), steering - 0.2
    return load_image(center), steering


def random_flip(image, steering):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering


def random_translate(image, steering, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering = steering + trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering


def augument(center, left, right, steering):
    image, steering = choose_image(center, left, right, steering)
    image, steering = random_flip(image, steering)
    image, steering = random_translate(image, steering, 100, 10)
    return image, steering


def batch_generator(X, y, batch_size, use_augumentation=False, steering_threshold=0.1, steering_prob=0.8):
    images = np.empty([batch_size, image_height, image_width, image_channels])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(X.shape[0]):
            center, left, right = X[index]
            steering = y[index]
            if use_augumentation:
                image, steering = augument(center, left, right, steering)
                while abs(steering) < steering_threshold and np.random.rand() < steering_prob:
                    image, steering = augument(center, left, right, steering)
            else:
                image = load_image(center)
            images[i] = preprocess(image)
            steers[i] = steering
            i += 1
            if i == batch_size:
                break
        yield images, steers

