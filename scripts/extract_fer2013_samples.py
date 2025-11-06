import pandas as pd
import numpy as np
import cv2
import os

csv_path = 'data/fer2013.csv'
save_folder = 'data/fer2013_samples/'
os.makedirs(save_folder, exist_ok=True)

def save_sample_image(df, idx, out_path):
    pixels = np.fromstring(df.loc[idx, 'pixels'], sep=' ').reshape(48, 48)
    cv2.imwrite(out_path, pixels.astype(np.uint8))

if __name__ == "__main__":
    df = pd.read_csv(csv_path)
    for i in range(3):
        out_path = f'{save_folder}/sample_{i}.png'
        save_sample_image(df, i, out_path)
        print('Saved:', out_path)
