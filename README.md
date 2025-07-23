# DEBRIS
The Database Exploration By Relative Image Similarity DEBRIS app allows a user to upload an image to retrieve related images from a chosen database. Currently, the images retrieved are from the MNIST fashion dataset, but future versions will allow the user to retrieve images from other datasets, including a personally uploaded dataset.

The retrieval algorithm uses a self-organizing map (an unsupervised machine learning algorithm) to categorize the images in the database based on the similarity of their visual features. To run the app, clone the repository and perform the following steps:

## Create Virtual Environment
Ensure you are in the "debris" directory. To create a virtual environment for the app using conda, run the following commands

### Unix/MacOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Windows:
```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

## Populate Database
```bash
python3 manage.py makemigrations
python3 manage.py migrate
python3 populate.py
```

## Run Server
```bash
python3 manage.py runserver
```

Once the app is running, use a browser to navigate to:

[http://localhost:8000/explore/](http://localhost:8000/explore/)
