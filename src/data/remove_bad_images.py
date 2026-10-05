import os
from PIL import Image

DATA_DIR = "data/processed/chest_xray"

bad_files = []

for root, dirs, files in os.walk(DATA_DIR):

    for file in files:

        if file.lower().endswith(".png"):

            path = os.path.join(root, file)

            try:
                with Image.open(path) as img:
                    img.verify()

            except Exception:
                bad_files.append(path)


print("Bad images found:", len(bad_files))

for path in bad_files:

    print("Removing:", path)
    os.remove(path)

print("\nCleanup completed!")