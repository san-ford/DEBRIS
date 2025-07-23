from PIL import Image
from scipy import ndimage
from sklearn_som.som import SOM
import numpy as np
import base64
import pickle
import io


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
    # double array to meet prediction requirements
    img = np.concatenate(([img], [img]))
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
def get_prediction(img):
    # prepare image for mapping
    loaded_model = pickle.load(open("model.pkl", "rb"))
    prediction = loaded_model.predict(img)
    return prediction[0]