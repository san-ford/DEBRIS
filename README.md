# DEBRIS
The Database Exploration By Relative Image Similarity (DEBRIS) app allows a user to browse a database by image similarity. Uploading an image will retrieve 10 similar images from a chosen database. The user can continue browsing by selecting one of the 10 images, which will retrieve another 10 images similar to the selection.

The user may choose to browse a custom database by uploading their own dataset of images. The images require no labeling or preprocessing, since an unsupervised machine learning algorithm is used to structure the databases. Two example databases of 10,000 images may also be browsed: Lanscapes and Fashion (built from an H&M catalogue).

When uploading a dataset of images, each object is run through a feature extraction pipeline to retrieve vector embeddings for each image. The embeddings are then organized by a self-organzing map, which structures the clusters of similar images in a way that facilitates more efficient retrieval. 

# Download
To download and run DEBRIS, ensure that your machine is utilizing Git LFS, then clone the repository.

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

# Using the App
The index page will give you three options for database exploration: two pre-populated database examples (Landscapes and Fashion), and a custom database, where the user creates a database by uploading their own images.

|![Index Page](https://github.com/user-attachments/assets/b0b81a92-39f3-41ce-86e5-de673fa4beb8)|
|:--:|
|**Index Page**|

Selecting one of the example databases redirects the user to the Image Query Upload page. When selecting the custom database option, the user is redirected to the Custom Database page (below), where they are met with another three options: upload images to populate the database, reorganize the self-organizing map to restructure the database, or upload an image to retrieve similar images from the current database. If the user has not yet populated the custom database, the last two options will have no effect.

|![Custom Database Page](https://github.com/user-attachments/assets/76b5b682-d8ab-4a90-bf5b-a914f0c86ca0)|
|:--:|
|**Custom Database Page**|

Database population may take a while, depending on how many images are uploaded. A counter is updated for every 100 images processed to inform the user of the upload progress. When the database population is complete, the self-organizing map begins structuring the database. Once restructured, the user is redirected to the Image Query Upload page.

|![Image Query Upload Page](https://github.com/user-attachments/assets/dd4724d5-8664-4f85-a749-a6bfe335175d)|
|:--:|
|**Image Query Upload Page**|

Uploading a single image query retrieves similar images from the database chosen at the Index page. The Results page shows seven images retrieved from the same node of the self-organizing map, displayed in order of visual similarity. The last three images are retrieved randomly from adjacent nodes.


|![Results Page](https://github.com/user-attachments/assets/cd7de930-524d-46c3-ba81-8a876c52ffe3)|
|:--:|
|**Results Page (from upload)**|

Any of the retrieved images can be selected as the next image query to retrieve another 10 images similar to the selection.

|![Results Page](https://github.com/user-attachments/assets/a7aa7443-1e94-4205-9ffd-0646107e3697)|
|:--:|
|**Results Page (from selection)**|

The user can then continue browsing the database by visual similarity. Additionally, the user may upload a new image query to restart the browsing process.
