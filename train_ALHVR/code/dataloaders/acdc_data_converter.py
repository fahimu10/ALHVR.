import glob
import os
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk

# Run this script from: train_ALHVR/code
RAW_DIR = Path("../../data/acdc/raw/training")
OUT_DIR = Path("../../data/acdc")
SLICE_DIR = OUT_DIR / "slices"

SLICE_DIR.mkdir(parents=True, exist_ok=True)

slice_num = 0

# Find all image files, but exclude ground-truth files
image_paths = sorted(glob.glob(str(RAW_DIR / "patient*" / "*_frame*.nii.gz")))
image_paths = [p for p in image_paths if "_gt" not in p]

print("Found image volumes:", len(image_paths))

for case in image_paths:
    img_itk = sitk.ReadImage(case)
    image = sitk.GetArrayFromImage(img_itk)

    msk_path = case.replace(".nii.gz", "_gt.nii.gz")

    if not os.path.exists(msk_path):
        print("Missing mask:", msk_path)
        continue

    print("Processing:", case)
    print("Mask:", msk_path)

    msk_itk = sitk.ReadImage(msk_path)
    mask = sitk.GetArrayFromImage(msk_itk)

    if image.shape != mask.shape:
        print("Shape mismatch:", image.shape, mask.shape)
        continue

    image = image.astype(np.float32)

    denom = image.max() - image.min()
    if denom > 0:
        image = (image - image.min()) / denom

    mask = mask.astype(np.uint8)

    item = Path(case).name.replace(".nii.gz", "")

    # Save full volume for validation/testing loader
    volume_path = OUT_DIR / f"{item}.h5"
    with h5py.File(volume_path, "w") as f:
        f.create_dataset("image", data=image, compression="gzip")
        f.create_dataset("label", data=mask, compression="gzip")

    # Save 2D training slices
    for slice_ind in range(image.shape[0]):
        slice_path = SLICE_DIR / f"{item}_slice_{slice_ind}.h5"

        with h5py.File(slice_path, "w") as f:
            f.create_dataset("image", data=image[slice_ind], compression="gzip")
            f.create_dataset("label", data=mask[slice_ind], compression="gzip")

        slice_num += 1

    print("Saved:", item, "slices:", image.shape[0])

print("Converted all ACDC volumes to 2D slices")
print("Total slices:", slice_num)

# =========================
# Regenerate data_list files
# =========================

data_list_dir = OUT_DIR / "data_list"
data_list_dir.mkdir(parents=True, exist_ok=True)

all_slices = sorted([p.stem for p in SLICE_DIR.glob("*.h5")])
all_volumes = sorted([p.stem for p in OUT_DIR.glob("patient*.h5")])

print("Generated slice files:", len(all_slices))
print("Generated volume files:", len(all_volumes))

# For local smoke test
train_slices = all_slices[:1312]
val_volumes = all_volumes[:20]

with open(data_list_dir / "train_slices.list", "w") as f:
    for item in train_slices:
        f.write(item + "\n")

with open(data_list_dir / "all_slices.list", "w") as f:
    for item in all_slices:
        f.write(item + "\n")

with open(data_list_dir / "train.list", "w") as f:
    for item in all_volumes:
        f.write(item + "\n")

with open(data_list_dir / "val.list", "w") as f:
    for item in val_volumes:
        f.write(item + "\n")

with open(data_list_dir / "test.list", "w") as f:
    for item in val_volumes:
        f.write(item + "\n")

print("Regenerated data_list files")
print("train_slices:", len(train_slices))
print("val:", len(val_volumes))