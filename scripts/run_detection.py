import os
import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

img_folder = 'data/fer2013_samples/'
img_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')]
detector = MTCNN()

for img_path in img_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Couldn't load image {img_path}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb, cmap='gray')
    plt.title(os.path.basename(img_path))
    plt.axis('off')
    plt.show(block=False)  
    plt.pause(2)           # Show each image for 2 seconds
    plt.close()
    
    faces = detector.detect_faces(img_rgb)
    print(f"Detected faces in {img_path}: {len(faces)}")
    # Draw bounding box
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.imshow(img_rgb, cmap='gray')
    plt.title('Detected Face(s)')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)           # Show each image for 2 seconds
    plt.close() 
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
        # If result is a list, take the first element
        if isinstance(result, list):
            result = result[0]

        print('Dominant emotion:', result['dominant_emotion'])
        print('Emotion scores:', result['emotion'])
    except Exception as e:
        print(f"DeepFace error for {img_path}: {e}")
