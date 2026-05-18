import os
from PIL import Image
import random

# 3 dummy classes just to test the pipeline
classes = ['class_a', 'class_b', 'class_c']
splits = ['train', 'val']

print("Generating dummy image dataset for testing...")

for split in splits:
    for cls in classes:
        # Create the directory structure: data/train/class_a, etc.
        path = os.path.join('data', split, cls)
        os.makedirs(path, exist_ok=True)
        
        # Generate 5 fake images per class
        for i in range(5):
            # Create a random solid color image (224x224)
            img_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new('RGB', (224, 224), color=img_color)
            
            # Save it
            img_name = f"dummy_{i}.jpg"
            img.save(os.path.join(path, img_name))

print("Done! Check your 'data/' folder.")