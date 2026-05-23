import os
import torch
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseServerError
from django.conf import settings
from .forms import ImageUploadForm
from .yolo_camera import generate_frames

image_model = None

def get_image_model():
    global image_model
    if image_model is None:
        image_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        image_model.conf = 0.40
    return image_model

def home(request):
    return render(request, 'detector/home.html')

def video_feed(request):
    try:
        return StreamingHttpResponse(
            generate_frames(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as error:
        return HttpResponseServerError(f"Camera error: {error}")

def image_detection(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            image = request.FILES['image']
            upload_path = os.path.join(settings.MEDIA_ROOT, image.name)

            with open(upload_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            model = get_image_model()
            results = model(upload_path)
            results.save(save_dir=settings.MEDIA_ROOT)

            detections = results.pandas().xyxy[0].to_dict(orient='records')

            return render(request, 'detector/image_result.html', {
                'image_url': settings.MEDIA_URL + image.name,
                'detections': detections,
            })
    else:
        form = ImageUploadForm()

    return render(request, 'detector/image_upload.html', {'form': form})
