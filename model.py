import cv2
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from pathlib import Path


DRIVING_LOG = Path(__file__).parent.joinpath("data", "data", "driving_log.csv")


def read_csv_file(filepath: Path):
    lines = []
    with DRIVING_LOG.open("r") as driving_log_file:
        reader = csv.DictReader(driving_log_file, delimiter=",")
        for line in reader:
            lines.append(line)

    return lines


def get_images_and_measurements():
    images = []
    measurements = []
    for line in lines:
        center_image_file = line["center"]
        filename = center_image_file.split("/")[-1]
        center_image_path = Path(__file__).parent.joinpath("data", "data", "IMG", filename)
        img = cv2.imread(str(center_image_path))
        images.append(img)
        measurements.append(float(line["steering"]))

    return images, measurements


def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    return model


lines = read_csv_file(DRIVING_LOG)
images, measurements = get_images_and_measurements()
X_train, y_train = np.array(images), np.array(measurements)

model = build_model()
model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save("model.h5")
