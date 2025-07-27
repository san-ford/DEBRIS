import os
import django
import numpy as np
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'debris.settings')
django.setup()

from explore.processing import encode_image
from explore.models import UploadedImages


# import images from csv file
images = pd.read_csv("debris/data/MNIST_fashion/fashion_train.csv")
# import list of image node predictions
predictions = pd.read_csv("debris/data/MNIST_fashion/fashion_prediction.csv")

# convert dataframes to numpy arrays
images = np.array(images)
predictions = np.array(predictions)

for i in range(len(images)):
    next_image = np.reshape(np.array(images[i]), (28, 28))
    image = UploadedImages(encoded_image=encode_image(next_image), node=predictions[i][0])
    image.save(using="MNIST_fashion")

print("Database populated successfully.")
