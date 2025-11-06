import os
import cv2
from mtcnn.mtcnn import MTCNN

# A. Gallagher, T. Chen, “Understanding Groups of Images of People,” IEEE Conference on Computer Vision and Pattern Recognition, 2009.

img_folder = 'data/group_images/'               # Input images with multiple faces
save_folder = 'data/cropped_faces/'             # Cropped face outputs
os.makedirs(save_folder, exist_ok=True)

face_detector = MTCNN()

for img_name in os.listdir(img_folder):
    if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
        continue
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(img_rgb)
    print(f"{img_name}: {len(faces)} face(s) detected")
    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        crop = img_rgb[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        face_resized = cv2.resize(crop, (48,48))     # Resize for model
        save_path = os.path.join(save_folder, f"{os.path.splitext(img_name)[0]}_face{idx}.png")
        cv2.imwrite(save_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")
