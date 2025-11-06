# group_facial_expression_recognition
Project to dectect facial expressions in group photos and videos along with integration of real time webcam 

# Recommend to create virtual environment 
python -m venv venv

# install dependencies
pip install -r requirements.txt

# inside the scripts folder
<!-- used for the detection of sample single image face expression -->
python run_detection.py 
<!-- Extracts few images as samples from the dataset fer2013 -->
python extract_fer2013_samples.py
<!-- Take input from data/group_images and crop each face in the image and resize for future use in data/cropped_images -->
python group_face_detection.py
<!-- Take input from data/cropped_images run face expression detection and aggregate the emotions -->
python group_face_detection.py