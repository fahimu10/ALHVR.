from pathlib import Path
import random

BASE = Path("../../data/acdc")
SLICE_DIR = BASE / "slices"
DATA_LIST = BASE / "data_list"

DATA_LIST.mkdir(parents=True, exist_ok=True)

all_slices = sorted([p.stem for p in SLICE_DIR.glob("*.h5")])
all_volumes = sorted([p.stem for p in BASE.glob("patient*.h5")])

print("Slices found:", len(all_slices))
print("Volumes found:", len(all_volumes))

random.seed(1337)
random.shuffle(all_slices)

# Keep repo-compatible sizes for smoke/reproduction code
train_slices = all_slices[:1312]
val_volumes = all_volumes[:20]

def write_list(path, items):
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\n")

write_list(DATA_LIST / "train_slices.list", train_slices)
write_list(DATA_LIST / "all_slices.list", all_slices)
write_list(DATA_LIST / "train.list", all_volumes)
write_list(DATA_LIST / "val.list", val_volumes)
write_list(DATA_LIST / "test.list", val_volumes)

print("Done.")
print("train_slices:", len(train_slices))
print("val:", len(val_volumes))