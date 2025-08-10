# DEBRIS
The Database Exploration By Relative Image Similarity (DEBRIS) app allows a user to browse a database by image similarity. Uploading an image will retrieve 10 similar images from a chosen database. The user can continue browsing by selecting one of the 10 images, which will retrieve another 10 images similar to the selection.

The user may choose to browse a custom database by uploading their own dataset of images. The images require no labeling or preprocessing, since an unsupervised machine learning algorithm is used to structure the databases. Two example databases may also be browsed: Lanscapes and Fashion (built from an H&M catalogue).

When uploading a dataset of images, each object is run through a feature extraction pipeline to retrieve vector embeddings for each image. The embeddings are then organized by a self-organzing map, which structures the clusters of similar images in a way that facilitates more efficient retrieval. 

## Create Virtual Environment
To create a virtual environment for the app using pip, run the following commands in the project directory:

#### Unix/MacOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

#### Windows:
```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

## Populate Database
#### Unix/MacOS:
```bash
python3 setup.py
```

#### Windows:
```bash
py setup.py
```

## Run Server
#### Unix/MacOS:
```bash
cd debris
python3 manage.py runserver
```

#### Windows:
```bash
cd debris
py manage.py runserver
```

Once the app is running, use a browser to navigate to:

[http://localhost:8000/explore/](http://localhost:8000/explore/)
