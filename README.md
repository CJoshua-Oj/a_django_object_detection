Django YOLOv5 Object Detection System

This project is a real-time object detection web application built using Django, YOLOv5, PyTorch, and OpenCV. It allows users to detect objects either through a live webcam stream or by uploading images directly through the web interface.

The application integrates Ultralytics YOLOv5 for accurate object recognition and uses Django as the backend framework to provide a clean web-based interface accessible through a browser.

Features
Real-time live webcam object detection
Image upload object detection
YOLOv5 pretrained object detection model
Bounding box visualization around detected objects
Confidence score display
Django web interface
OpenCV webcam integration
Browser-based live video stream
Responsive frontend UI
Easy VS Code setup and execution

Technologies Used

Python
Django
YOLOv5
PyTorch
OpenCV
HTML
CSS
JavaScript
SQLite

Project Modules

Live Webcam Detection
Uses the computer webcam
Detects objects in real time
Streams detection output to the browser

Image Upload Detection

Allows users to upload images
Detects multiple objects in uploaded images
Displays detected objects with confidence scores

Installation

git clone YOUR_GITHUB_REPOSITORY_LINK
cd django-object-detection-master
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

Run Application

Open browser:

http://127.0.0.1:8000/

Important Notes

Internet connection is required during first run to download YOLOv5 pretrained weights.
Ensure webcam permission is enabled.
Close other applications using the webcam before starting.
Developed for VS Code local execution.

Use Cases

Smart surveillance systems
Security monitoring
Educational computer vision projects
AI object recognition demonstrations
Research and machine learning experimentation

Author

C.Joshua O. / CJoshua-Oj 
