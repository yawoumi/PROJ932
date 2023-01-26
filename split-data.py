import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

image_dir = "./all_img/"

os.makedirs("train")
os.makedirs("test")
os.makedirs("validation")

# Get a list of all the images in the directory
images = os.listdir(image_dir)

# Split the images into train, test, and validation sets
train_images, test_val_images, _, _ = train_test_split(images, images, test_size=0.3, random_state=42)
val_images, test_images, _, _ = train_test_split(test_val_images, test_val_images, test_size=0.5, random_state=42)

# Move the images into the corresponding directories
for image in train_images:
    os.rename(os.path.join(image_dir, image), os.path.join("./train/", image))
for image in test_images:
    os.rename(os.path.join(image_dir, image), os.path.join("./test/", image))
for image in val_images:
    os.rename(os.path.join(image_dir, image), os.path.join("./validation/", image))