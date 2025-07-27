from PIL import Image
import os

def clean_dataset(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                img = Image.open(filepath)
                img.verify()  # Check if image can be opened
            except Exception as e:
                print(f"❌ Removing corrupted image: {filepath}")
                os.remove(filepath)

clean_dataset("dataset")  # path to your dataset root folder
