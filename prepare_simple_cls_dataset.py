# prepare_simple_cls_dataset.py
import os
import shutil
from tqdm import tqdm

def prepare_split(src_split, dst_split):
    os.makedirs(os.path.join(dst_split, "solar"), exist_ok=True)
    os.makedirs(os.path.join(dst_split, "no_solar"), exist_ok=True)

    img_dir = os.path.join(src_split, "images")
    lbl_dir = os.path.join(src_split, "labels")

    if not os.path.exists(img_dir):
        print(f"Missing {img_dir}. Skipping.")
        return

    for img_name in tqdm(os.listdir(img_dir), desc=f"Processing {src_split}"):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        # class rule: any non-empty label file -> solar
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            cls = "solar"
        else:
            cls = "no_solar"
        src_img = os.path.join(img_dir, img_name)
        dst_img = os.path.join(dst_split, cls, img_name)
        shutil.copy2(src_img, dst_img)

if __name__ == "__main__":
    # adjust if your folders are named differently
    splits = [("train", "simple_dataset/train"), ("valid", "simple_dataset/valid")]
    for src, dst in splits:
        prepare_split(src, dst)
    print("Done. Created simple_dataset with train/valid splits.")
