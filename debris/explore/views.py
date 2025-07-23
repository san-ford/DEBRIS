from django.shortcuts import render
from django.db.models import Q
from .models import ImageSubmitted, ImageRetrieved
from .processing import encode_image, decode_image, preprocess, get_prediction
import numpy as np
from PIL import Image
import random


def index(request):
    return render(request, "explore/index.html")


def result(request):
    # verify POST request
    if request.method == 'POST':
        # if image was selected from list
        if not request.FILES:
            # retrieve content selected
            content = request.POST.dict()
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
                    image_submitted.node = get_prediction(processed_image)
            else:
                context = {"error_message": "Invalid selection."}
                return render(request, "explore/index.html", context)
        # if file was uploaded
        elif request.FILES["image"]:
            image_submitted = request.FILES["image"]
            # specify allowed file types
            allowed_file_types = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
            # verify uploaded file is allowed type
            if image_submitted.content_type not in allowed_file_types:
                context = {"error_message": "Invalid file type."}
                return render(request, "explore/index.html", context)
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
            # retrieve prediction from submitted image
            image_submitted.node = get_prediction(processed_image)

        # error if no file uploaded
        else:
            context = {"error_message": "No file selected."}
            return render(request, "explore/index.html", context)

        # retrieve images from database related to submitted image
        retrieved_images = ImageRetrieved.objects.filter(node__exact=image_submitted.node)
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
        neighbor_images = ImageRetrieved.objects.filter(Q(node__exact=neighbor_nodes[0]) |
                                                        Q(node__exact=neighbor_nodes[1]) |
                                                        Q(node__exact=neighbor_nodes[2]) |
                                                        Q(node__exact=neighbor_nodes[3]))

        # determine number of neighbor images to retrieve
        neighbor_sample_size = min(10 - node_sample_size, len(neighbor_images))

        # retrieve neighboring images
        for i in range(neighbor_sample_size):
            candidate = random.choice(neighbor_images)
            while candidate in related_images:
                candidate = random.choice(neighbor_images)
            related_images.append(candidate)

        context = {"image_submitted": image_submitted, "related_images": related_images}
        return render(request, "explore/result.html", context)

    # return to index if no POST request
    else:
        return render(request, "explore/index.html")
