import os
from PIL import Image

DATA_DIR = "data/processed/chest_xray"

bad_files = []
total = 0

for root, dirs, files in os.walk(DATA_DIR):

    for file in files:

        if file.lower().endswith(".png"):

            path = os.path.join(root, file)
            total += 1

            try:
                with Image.open(path) as img:
                    img.verify()

            except Exception as e:
                bad_files.append((path, str(e)))

print("\n==============================")
print("IMAGE CHECK COMPLETED")
print("==============================")

print("Total PNG files:", total)
print("Corrupted files:", len(bad_files))

if bad_files:

    print("\nCorrupted files:")

    for path, error in bad_files:
        print(path)
        print("Error:", error)

else:
    print("\nAll PNG files are valid!")