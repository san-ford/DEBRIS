from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .processing import preprocess, clear_db, populate_db, create_som, retrieve, create_model
import json


@csrf_exempt
def index(request):
    return render(request, "explore/index.html")


@csrf_exempt
def create_db(request):
    # verify post request
    if request.method == 'POST':
        # retrieve batch of files
        if request.FILES:
            uploaded_files = request.FILES.getlist('files')

        if 'som' in request.POST.dict():
            status = create_som()
            return JsonResponse({"som_status": status})

        # clear default database and initialize ML model on first run
        if 'file_number' in request.POST.dict():
            if request.POST.get('file_number') == '0':
                clear_db()
                create_model()

        # populate database with batch
        rejected = populate_db(uploaded_files)
        return JsonResponse({'status': 'chunk_received',
                             'files_rejected': rejected})
    return render(request, "explore/create_db.html")


@csrf_exempt
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
        # if self-organizing map needs to be created
        elif "som" in request.POST.dict():
            som = create_som()
            context = {"database": "default", "som": som}
            return render(request, "explore/upload.html", context)
        else:
            som = "SOM not reorganized"
            context = {"database": "default", "som": som}
            return render(request, "explore/upload.html", context)
    # if no POST requested, use default database
    else:
        context = {"database": "default"}
        return render(request, "explore/upload.html", context)


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
                encoded_image = content["selection"]
                # retrieve node from selected image
                if "node" in content and "embeddings" in content:
                    node = json.loads(content["node"])
                    embeddings = json.loads(content["embeddings"])
                # error if the node is not retrieved
                else:
                    context = {"error_message": "Invalid selection."}
                    return render(request, "explore/index.html", context)
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
            # retrieve image embeddings, encoding, and predicted node
            embeddings, encoded_image, node = preprocess(image_submitted, database)

        # error if no file uploaded
        else:
            context = {"error_message": "No file selected."}
            return render(request, "explore/index.html", context)

        # retrieve images from database related to submitted image
        related_images = retrieve(embeddings, node, database)

        context = {
            "image_submitted": encoded_image,
            "related_images": related_images,
            "database": database
        }
        return render(request, "explore/result.html", context)

    # return to index if no POST request
    else:
        return render(request, "explore/index.html")
