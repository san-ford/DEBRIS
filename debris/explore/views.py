from django.shortcuts import render
from django.db.models import Q
from .models import ImageSubmitted, UploadedImages
from .processing import (encode_image, decode_image, preprocess,
                         get_prediction, populate_db, clear_db)
import numpy as np
from PIL import Image
import random


def index(request):
    return render(request, "explore/index.html")


def create_db(request):
    # verify POST request
    if request.method == 'POST':
        # verify upload path provided
        if 'upload_path' in request.POST:
            # clear existing database
            clear_db()
            path = request.POST['upload_path']
            # populate database
            acc, rej, t, error_message = populate_db(path)
            # input verification
            if error_message:
                context = {"error_message": error_message}
                return render(request, "explore/create_db.html", context)
            # if database population was successful
            context = {"database": "default",
                       "accepted": acc,
                       "rejected": rej,
                       "time": t}
            return render(request, "explore/upload.html", context)
        else:
            context = {"error_message": "No directory selected"}
            return render(request, "explore/create_db.html", context)
    else:
        context = {"error_message": "No directory selected"}
        return render(request, "explore/create_db.html", context)


def upload(request):
    # verify POST request
    if request.method == 'POST':
        # retrieve content selected
        content = request.POST.dict()
        # verify database is specified
        if "database" in content:
            database = content["database"]
            # if custom database needs to be populated, go to create db view
            if database == "default" and "db_created" not in content:
                return render(request, "explore/create_db.html")
            # otherwise, prompt user to upload an image query
            context = {"database": database}
            return render(request, "explore/upload.html", context)
        else:
            context = {"error_message": "No database specified"}
            return render(request, "explore/index.html", context)
    # return to index if no POST request
    else:
        return render(request, "explore/index.html")


def result(request):
    # verify POST request
    if request.method == 'POST':
        # if image was selected from list (no file uploaded)
        if not request.FILES:
            # retrieve content selected
            content = request.POST.dict()
            # verify database is specified
            if "database" in content:
                database = content["database"]
            # error if no database specified
            else:
                context = {"error_message": "No database specified"}
                return render(request, "explore/index.html", context)
            # verify selection is valid
            if "selection" in content:
                # define image submitted as image selected (already encoded)
                image_submitted = ImageSubmitted(submission=content["selection"])
                # retrieve node from selected image
                if "node" in content:
                    image_submitted.node = int(content["node"])
                # if the node is not retrieved, predict it
                else:
                    processed_image = decode_image(image_submitted)
                    # flatten image array
                    processed_image = np.reshape(processed_image, 784)
                    # double array to meet prediction requirements
                    processed_image = np.concatenate(([processed_image], [processed_image]))
                    # retrieve prediction from submitted image
                    image_submitted.node = get_prediction(processed_image, database)
            else:
                context = {"error_message": "Invalid selection."}
                return render(request, "explore/index.html", context)
        # if file was uploaded
        elif "image" in request.FILES:
            # verify database is valid
            database = request.POST.dict()
            if "database" in database:
                database = database["database"]
            else:
                context = {"error_message": "No database selected"}
                return render(request, "explore/index.html", context)
            image_submitted = request.FILES["image"]
            # specify allowed file types
            allowed_file_types = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
            # verify uploaded file is allowed type
            if image_submitted.content_type not in allowed_file_types:
                context = {"error_message": "Invalid file type."}
                return render(request, "explore/upload.html", context)
            # retrieve submitted image
            image_submitted = Image.open(image_submitted)
            # create greyscale version of submitted image for preprocessing
            processed_image = image_submitted.convert("L")
            # prepare image for result view
            image_submitted = encode_image(np.array(image_submitted))
            # create instance of image submitted
            image_submitted = ImageSubmitted(submission=image_submitted)
            # preprocess image for ML prediction
            processed_image = preprocess(processed_image)
            # double array to meet prediction requirements
            processed_image = np.concatenate(([processed_image], [processed_image]))
            # retrieve prediction from submitted image
            image_submitted.node = get_prediction(processed_image, database)

        # error if no file uploaded
        else:
            context = {"error_message": "No file selected."}
            return render(request, "explore/index.html", context)

        # retrieve images from database related to submitted image
        retrieved_images = UploadedImages.objects.using(database).filter(node__exact=image_submitted.node)
        node_sample_size = min(7, len(retrieved_images))
        related_images = []
        for i in range(node_sample_size):
            candidate = random.choice(retrieved_images)
            while candidate in related_images:
                candidate = random.choice(retrieved_images)
            related_images.append(candidate)

        # determine neighboring nodes
        neighbor_nodes = [-1]*4
        node = int(image_submitted.node)
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
        neighbor_sample_size = min(10 - node_sample_size, len(neighbor_images))

        # retrieve neighboring images
        for i in range(neighbor_sample_size):
            candidate = random.choice(neighbor_images)
            while candidate in related_images:
                candidate = random.choice(neighbor_images)
            related_images.append(candidate)

        context = {
            "image_submitted": image_submitted,
            "related_images": related_images,
            "database": database
        }
        return render(request, "explore/result.html", context)

    # return to index if no POST request
    else:
        return render(request, "explore/index.html")
