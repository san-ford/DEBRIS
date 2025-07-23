import os
import django
import numpy as np
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'debris.settings')
django.setup()

from explore.processing import encode_image, decode_image, preprocess, get_prediction
from explore.models import ImageSubmitted, ImageRetrieved


# import images from csv file
images = pd.read_csv("train.csv")
# import list of image node predictions
predictions = pd.read_csv("fashion_prediction.csv")

# convert dataframes to numpy arrays
images = np.array(images)
predictions = np.array(predictions)

for i in range(len(images)):
    next_image = np.reshape(np.array(images[i]), (28, 28))
    image = ImageRetrieved(encoded_image=encode_image(next_image), node=predictions[i][0])
    image.save()

print("Database populated successfully.")