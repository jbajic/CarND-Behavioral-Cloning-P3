import cv2
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

MODEL_NAME = "model.h5"
BATCH_SIZE = 128
EPOCHS = 15
Dataset = namedtuple("Dataset", ("driving_log", "data_path"))

DATASETS = [
    Dataset("data/driving_log.csv", "data/IMG"),
    Dataset("data2/driving_log.csv", "data2/IMG"),
]


def read_lines(filepath):
    lines = []
    with open(filepath, "r") as driving_log_file:
        reader = csv.DictReader(driving_log_file, delimiter=",")
        for line in reader:
            lines.append(line)
    return lines


def get_image_path(data_path, image_name):
    return os.path.join(data_path, image_name)


def get_image_steering_correction(side):
    if side == "right":
        return -0.2
    elif side == "left":
        return 0.2
    elif side == "center":
        return 0


def get_images_and_measurements(lines):
    images = []
    measurements = []
    for line in lines:
        for camera_side in ["center", "left", "right"]:
            image_file = line[camera_side]

            filename = image_file.split("/")[-1]
            image_path = get_image_path(line["data_path"], filename)

            img = cv2.imread(str(image_path))
            images.append(img)

            correction = get_image_steering_correction(camera_side)
            steering_center = float(line["steering"]) + correction
            measurements.append(steering_center)

    return images, measurements


def augment_data(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1)

    return augmented_images, augmented_measurements


def image_generator(lines):
    while True:
        lines = shuffle(lines)
        for batch_index in range(0, len(lines), BATCH_SIZE):
            batch_data = lines[batch_index : batch_index + BATCH_SIZE]
            images = []
            measurements = []
            images, measurements = get_images_and_measurements(batch_data)
            X_train, y_train = np.array(images), np.array(measurements)

            yield shuffle(X_train, y_train)


def build_model():
    model = Sequential()
    # Preprocessing
    # Standardization
    model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160, 320, 3)))
    # Crop image
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
    # model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
    # model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))

    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="relu"))

    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    return model


lines = []
for dataset in DATASETS:
    print("Reading dataset: {dataset}".format(dataset=dataset.data_path))
    new_lines = read_lines(dataset.driving_log)
    for new_line in new_lines:
        new_line["data_path"] = dataset.data_path
    lines.extend(new_lines)

training_dataset, validation_dataset = train_test_split(lines, test_size=0.2)

# images, measurements = [], []
# new_images, new_measurements = get_images_and_measurements(lines, dataset.data_path)
#    print("Dataset contains {num} images".format(num=len(new_images)))
#     augumented_images, augumented_measurements = augument_data(new_images, new_measurements)
#     images.extend(augumented_images)
#     measurements.extend(augumented_measurements)

# X_train, y_train = np.array(images), np.array(measurements)

number_of_train_samples = len(training_dataset)
number_of_validation_samples = len(validation_dataset)
model = build_model()
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# history_object = model.fit_generator(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15, verbose=2)
history_object = model.fit_generator(
    image_generator(training_dataset),
    validation_data=image_generator(validation_dataset),
    epochs=EPOCHS,
    verbose=2,
    steps_per_epoch=np.ceil(number_of_train_samples / BATCH_SIZE),
    validation_steps=np.ceil(number_of_validation_samples / BATCH_SIZE),
)
model.save(MODEL_NAME)
print("Model {model} saved!".format(model=MODEL_NAME))

### plot the training and validation loss for each epoch
plt.plot(history_object.history["loss"])
plt.plot(history_object.history["val_loss"])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.legend(["training set", "validation set"], loc="upper right")
# plt.show()
plt.savefig("validation_loss_diagram.png")
