import os
import zipfile
import shutil
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# =========================
# PATHS
# =========================

BASE_DIR = "data/raw/chest_xray"

ZIP_PATH = os.path.join(
    BASE_DIR,
    "stage_2_train_images.zip"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "stage_2_train_labels.csv"
)

EXTRACT_DIR = os.path.join(
    BASE_DIR,
    "dicom_images"
)

OUTPUT_DIR = "data/processed/chest_xray"


# =========================
# CREATE DIRECTORIES
# =========================

os.makedirs(EXTRACT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, label),
            exist_ok=True
        )


# =========================
# STEP 1: EXTRACT ZIP
# =========================

print("\n[1/5] Checking DICOM files...")

if not os.listdir(EXTRACT_DIR):

    print("Extracting ZIP...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print("ZIP extraction completed.")

else:
    print("DICOM files already extracted.")


# =========================
# FIND DICOM DIRECTORY
# =========================

dicom_root = EXTRACT_DIR

# Handle ZIP containing stage_2_train_images folder
possible_folder = os.path.join(
    EXTRACT_DIR,
    "stage_2_train_images"
)

if os.path.exists(possible_folder):
    dicom_root = possible_folder


# =========================
# STEP 2: LOAD LABELS
# =========================

print("\n[2/5] Loading labels...")

df = pd.read_csv(LABEL_PATH)

print("Total rows:", len(df))

# One patient can have multiple bounding boxes.
# Patient-level label:
# If ANY row has Target = 1 → PNEUMONIA
patient_labels = (
    df.groupby("patientId")["Target"]
    .max()
    .reset_index()
)

patient_labels["label"] = patient_labels["Target"].map({
    0: "NORMAL",
    1: "PNEUMONIA"
})

print("\nClass distribution:")
print(patient_labels["label"].value_counts())


# =========================
# STEP 3: TRAIN / VAL / TEST
# =========================

print("\n[3/5] Creating patient-level splits...")

train_df, temp_df = train_test_split(
    patient_labels,
    test_size=0.20,
    random_state=42,
    stratify=patient_labels["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# =========================
# STEP 4: CONVERT DICOM → PNG
# =========================

def convert_dicom_to_png(patient_id, output_path):

    dicom_path = os.path.join(
        dicom_root,
        patient_id + ".dcm"
    )

    if not os.path.exists(dicom_path):
        return False

    try:

        ds = pydicom.dcmread(dicom_path)

        image = ds.pixel_array.astype(np.float32)

        # Normalize pixel values
        image -= image.min()

        if image.max() > 0:
            image /= image.max()

        image = (image * 255).astype(np.uint8)

        # Convert to PIL image
        pil_image = Image.fromarray(image)

        # Convert grayscale → RGB
        pil_image = pil_image.convert("RGB")

        # Save PNG
        pil_image.save(output_path)

        return True

    except Exception as e:

        print(
            f"Error processing {patient_id}: {e}"
        )

        return False


# =========================
# PROCESS SPLITS
# =========================

def process_split(dataframe, split_name):

    print(
        f"\nProcessing {split_name}..."
    )

    total = len(dataframe)
    success = 0

    for index, row in dataframe.iterrows():

        patient_id = row["patientId"]
        label = row["label"]

        output_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            label
        )

        output_file = os.path.join(
            output_dir,
            patient_id + ".png"
        )

        # Skip existing files
        if os.path.exists(output_file):
            success += 1
            continue

        result = convert_dicom_to_png(
            patient_id,
            output_file
        )

        if result:
            success += 1

        # Progress
        if success % 500 == 0:
            print(
                f"Processed {success}/{total}"
            )

    print(
        f"{split_name} completed: "
        f"{success}/{total}"
    )


# =========================
# STEP 5: PROCESS DATASET
# =========================

print("\n[4/5] Converting images...")

process_split(train_df, "train")
process_split(val_df, "val")
process_split(test_df, "test")


# =========================
# SUMMARY
# =========================

print("\n[5/5] Dataset preparation completed!")

print("\nDataset location:")
print(OUTPUT_DIR)

for split in ["train", "val", "test"]:

    for label in ["NORMAL", "PNEUMONIA"]:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            label
        )

        count = len([
            f for f in os.listdir(folder)
            if f.endswith(".png")
        ])

        print(
            f"{split}/{label}: {count}"
        )

print("\nREADY FOR TRAINING!")