import os
import cv2
from deepface import DeepFace
from collections import Counter

face_folder = 'data/cropped_faces/'
results = []                                         

face_files = [f for f in os.listdir(face_folder) if f.endswith('.png')]
image_groups = {}
# Group face files by their base image name
for face_file in face_files:
    base = face_file.split('_face')[0]
    image_groups.setdefault(base, []).append(face_file)

for base_img, group_faces in image_groups.items():
    emotions = []
    for face_file in group_faces:
        img_path = os.path.join(face_folder, face_file)
        try:
            result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            emotions.append(result['dominant_emotion'])
        except Exception as e:
            print(f"Error analyzing {face_file}: {e}")
            continue
    if emotions:
        group_emotion = Counter(emotions).most_common(1)[0][0]
        print(f"Group '{base_img}': {group_emotion} (faces: {emotions})")
        results.append((base_img, group_emotion, emotions))
    else:
        print(f"No valid faces/emotions found for {base_img}")


import csv
with open('results/group_emotions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['base_image', 'group_emotion', 'individual_emotions'])
    for record in results:
        writer.writerow([record[0], record[1], str(record[2])])
print("Saved group-level emotion results to 'results/group_emotions.csv'")
