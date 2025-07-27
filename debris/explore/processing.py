from PIL import Image
from scipy import ndimage
from sklearn_som.som import SOM
import numpy as np
import subprocess
import django
import base64
import pickle
import time
import sys
import io
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'debris.settings')
django.setup()

from .models import UploadedImages


def preprocess(img):
    # edge detection (subtract Gaussian blurred image from original)
    img = img - ndimage.gaussian_filter(img, 3)
    # center object
    img = center_object(img)
    # convert back to Image object
    img = Image.fromarray(img)

    # make image square
    w, h = img.size
    if w > h:
        h = w
    else:
        w = h
    img = img.resize((w, h))
    # scale image
    img.thumbnail((28, 28))
    img = np.array(img)
    # flatten image array
    img = np.reshape(img, 784)
    return img


def center_object(img):
    # find center of mass
    y_center, x_center = ndimage.center_of_mass(img)
    y_center = int(y_center)
    x_center = int(x_center)

    # determine distances from center of mass to edge of image
    y_dist_to_edge = min(y_center, len(img) - y_center)
    x_dist_to_edge = min(x_center, len(img[0]) - x_center)
    dist_to_edge = max(y_dist_to_edge, x_dist_to_edge)

    # zero padding
    if x_dist_to_edge > y_dist_to_edge:
        # if overhang is on top
        if y_center - x_dist_to_edge < 0:
            overhang = x_dist_to_edge - y_center
            # zero pad on top
            img = np.append([[0] * len(img[0])] * overhang, img, axis=0)
            # move y center after padding on top
            y_center += overhang
        # else overhang is on bottom
        else:
            overhang = x_dist_to_edge - y_dist_to_edge
            # zero pad on bottom
            img = np.append(img, [[0] * len(img[0])] * overhang, axis=0)
    if y_dist_to_edge > x_dist_to_edge:
        # if overhang is on left
        if x_center - y_dist_to_edge < 0:
            overhang = y_dist_to_edge - x_center
            # zero pad on left
            img = np.append([[0] * overhang] * len(img), img, axis=1)
            # move x center after padding on left
            x_center += overhang
        # else overhang is on right
        else:
            overhang = y_dist_to_edge - x_dist_to_edge
            # zero pad on right
            img = np.append(img, [[0] * overhang] * len(img), axis=1)

    # crop image as square
    img = img[y_center - dist_to_edge:y_center + dist_to_edge, x_center - dist_to_edge:x_center + dist_to_edge]
    return img.astype(np.uint8)


def encode_image(img):
    img = Image.fromarray(img.astype("uint8"))
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, 'PNG')
    encoded_image = "data:image/png;base64," + base64.b64encode(raw_bytes.getvalue()).decode('ascii')
    return encoded_image


def decode_image(img):
    base64_decoded = base64.b64decode(img[21:])
    decoded_image = Image.open(io.BytesIO(base64_decoded))
    decoded_image = np.array(decoded_image)
    return decoded_image


# retrieve prediction from saved model
def get_prediction(img, db):
    # prepare image for mapping
    loaded_model = pickle.load(open("data/{}/{}.pkl".format(db, db), "rb"))
    prediction = loaded_model.predict(img)
    return prediction[0]


def populate_db(path):
    start = time.time()
    # validate directory path
    if not os.path.exists(path):
        return 0, 0, 0, "Invalid path file"
    images = []
    processed_images = []
    files_accepted = 0
    files_rejected = 0
    time_elapsed = 0
    allowed_file_types = ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']

    # retrieve and process files
    for file in os.scandir(path):
        # verify path leads to a file
        if file.is_file():
            # verify valid file type
            _, extension = os.path.splitext(file)
            if extension.lower() not in allowed_file_types:
                files_rejected += 1
                continue
            files_accepted += 1
            # import each image
            img = Image.open(file.path)
            # resize image
            w, h = img.size
            ratio = 140 / w
            size = (140, int(h*ratio))
            # store image as numpy array
            img = img.resize(size)
            # store each encoded image (with color)
            images.append(encode_image(np.array(img)))
            # store each processed image (convert to greyscale)
            processed_images.append(preprocess(img.convert("L")))

    if not files_accepted:
        message = "No valid files in directory"
        return files_accepted, files_rejected, time_elapsed, message

    # prepare array for self-organizing map
    processed_images = np.array(processed_images)
    # initiate a 10x10 self-organizing map with input dimensions = 784
    custom_som = SOM(m=10, n=10, dim=784)
    custom_som.fit(processed_images)
    # transform the map to organize the training data
    custom_map = custom_som.transform(processed_images)
    # Save model using Pickle
    with open("data/default/default.pkl", "wb") as model_file:
        pickle.dump(custom_som, model_file)
    # find the closest node for each data point
    nodes = custom_som.predict(processed_images)

    # save encoded images and predicted nodes
    for i in range(len(images)):
        next_image = UploadedImages(encoded_image=images[i], node=nodes[i])
        next_image.save()

    message = None
    time_elapsed = (time.time() - start) / 60

    return files_accepted, files_rejected, time_elapsed, message


def clear_db():
    if sys.platform == "win32":
        cmd = "py"
    else:
        cmd = "python3"

    subprocess.run([cmd, "manage.py", "flush", "--noinput"])
