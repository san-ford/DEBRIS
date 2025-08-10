from transformers import AutoImageProcessor, AutoModel
from accelerate.test_utils.testing import get_backend
from torch.nn.functional import cosine_similarity
from sklearn_som.som import SOM
from django.db.models import Q
from PIL import Image
import numpy as np
import django, base64, pickle
import sys, io, os
import subprocess
import random
import torch

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'debris.settings')
django.setup()

from .models import UploadedImages


def preprocess(image, db):
    # prepare feature extraction variables
    DEVICE, _, _ = get_backend()
    # retrieve pre-loaded models
    processor = pickle.load(open("data/{}/processor.pkl".format(db), "rb"))
    model = pickle.load(open("data/{}/model.pkl".format(db), "rb"))

    # retrieve image
    img = Image.open(image).convert("RGB")
    encoded_image = encode_image(np.array(img))
    # create embeddings for image
    emb = processor(img, return_tensors="pt").to(DEVICE)
    emb = model.to(DEVICE)(**emb).pooler_output
    # store image embeddings as numpy array
    embeddings = emb.cpu().detach().numpy()[0]

    # retrieve node prediction
    node = get_prediction(embeddings, db)

    return embeddings, encoded_image, node


# encode image for HTML view
def encode_image(img):
    img = Image.fromarray(img.astype("uint8"))
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, 'PNG')
    encoded_image = "data:image/png;base64," + base64.b64encode(raw_bytes.getvalue()).decode('ascii')
    return encoded_image


# retrieve prediction from saved model
def get_prediction(img, db):
    # double array to meet prediction algorithm requirements
    img = np.concatenate(([img], [img]))
    # prepare image for mapping
    loaded_model = pickle.load(open("data/{}/{}.pkl".format(db, db), "rb"))
    prediction = loaded_model.predict(img)
    return prediction[0]


def clear_db(database="default"):
    if sys.platform == "win32":
        cmd = "py"
    else:
        cmd = "python3"

    subprocess.run([cmd, "manage.py", "flush", "--noinput", "--database={}".format(database)])

    return


def populate_db(files, database="default"):
    allowed_file_types = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']

    # prepare feature extraction variables
    DEVICE, _, _ = get_backend()
    # retrieve pre-loaded models
    processor = pickle.load(open("data/{}/processor.pkl".format(database), "rb"))
    model = pickle.load(open("data/{}/model.pkl".format(database), "rb"))

    # keep track of rejected files
    rejected = 0

    # retrieve each file
    for f in files:
        # verify image type
        if f.content_type not in allowed_file_types:
            rejected += 1
            continue

        # retrieve image file
        img = Image.open(f).convert("RGB")
        # create embeddings for image
        emb = processor(img, return_tensors="pt").to(DEVICE)
        emb = model.to(DEVICE)(**emb).pooler_output
        # convert embeddings to python list
        emb = emb.tolist()[0]
        # resize image
        w, h = img.size
        ratio = 200 / w
        size = (200, int(h*ratio))
        # store image as numpy array
        img = img.resize(size)
        # save encoded image and embeddings to database
        next_image = UploadedImages(encoded_image=encode_image(np.array(img)),
                                    embeddings=emb)
        next_image.save(using=database)

    return rejected


def create_som(database="default"):
    # retrieve embeddings from database
    embeddings = []
    images = UploadedImages.objects.using(database).all()
    # handle empty database
    if not len(images):
        return "No data to create SOM"
    for i in images:
        embeddings.append(i.embeddings)
    embeddings = np.array(embeddings)
    # initiate a 10x10 self-organizing map with input dimensions = 768
    custom_som = SOM(m=10, n=10, dim=768)
    custom_som.fit(embeddings)
    # transform the map to organize the training data
    custom_map = custom_som.transform(embeddings)
    # Save model using Pickle
    with open("data/{}/{}.pkl".format(database, database), "wb") as model_file:
        pickle.dump(custom_som, model_file)
    # find the closest node for each data point
    nodes = custom_som.predict(embeddings)

    # save predicted nodes to database
    for i in range(len(images)):
        next_image = images[i]
        next_image.node = nodes[i]
        next_image.save(using=database)
    return "SOM created"


def retrieve(embeddings, node, database):
    embeddings = np.array(embeddings)
    # retrieve images from database related to submitted image
    retrieved_images = UploadedImages.objects.using(database).filter(node__exact=node)
    # determine similarity score for each retrieved image
    ranked_images = []
    for i in range(len(retrieved_images)):
        similarity_score = cosine_similarity(torch.Tensor(embeddings.tolist()),
                                             torch.Tensor(retrieved_images[i].embeddings), dim=0)
        ranked_images.append([retrieved_images[i], float(similarity_score)])

    # sort images by highest ranking
    ranked_images = sorted(ranked_images, key=lambda x: x[1], reverse=True)
    # top image is the same image, remove it
    if len(ranked_images) and ranked_images[0][1] >= 0.99:
        ranked_images.pop(0)

    # specify number up to 7 of the top images to display
    node_sample_size = min(7, len(ranked_images))
    related_images = []
    for i in range(node_sample_size):
        related_images.append(ranked_images[i][0])

    # determine neighboring nodes
    neighbor_nodes = [-1] * 4
    if node > 10:
        neighbor_nodes[0] = node - 10
    if node < 91:
        neighbor_nodes[1] = node + 10
    if node % 10 != 1:
        neighbor_nodes[2] = node - 1
    if node % 10 != 0:
        neighbor_nodes[3] = node + 1

    # retrieve images from neighboring nodes
    neighbor_images = UploadedImages.objects.using(database).filter(
        Q(node__exact=neighbor_nodes[0]) |
        Q(node__exact=neighbor_nodes[1]) |
        Q(node__exact=neighbor_nodes[2]) |
        Q(node__exact=neighbor_nodes[3])
    )

    # determine number of neighbor images to retrieve
    neighbor_sample_size = min(10 - len(related_images), len(neighbor_images))

    # retrieve neighboring images
    for i in range(neighbor_sample_size):
        candidate = random.choice(neighbor_images)
        while candidate in related_images:
            candidate = random.choice(neighbor_images)
        related_images.append(candidate)

    return related_images


def create_model(database="default"):
    # prepare feature extraction variables
    DEVICE, _, _ = get_backend()
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModel.from_pretrained("google/vit-base-patch16-224")

    with open("data/{}/processor.pkl".format(database), "wb") as model_file:
        pickle.dump(processor, model_file)

    with open("data/{}/model.pkl".format(database), "wb") as model_file:
        pickle.dump(model, model_file)
    return