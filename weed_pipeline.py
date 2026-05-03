"""
Gorev 4 — Yabanci Ot (Weed) Tespit Pipeline

Adimlar:
  0. extract   — archive.zip → weed_data/
  1. convert   — Pascal VOC XML → YOLO txt
  2. split     — 70 / 15 / 15 random split
  3. yaml      — weed_yolo_dataset/data.yaml
  4. train     — YOLOv8m egitimi

Kullanim:
    conda activate crop_detection
    python weed_pipeline.py
    python weed_pipeline.py --skip-extract   # zip zaten cikartildiysa
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from weed_config import (
    AGRI_DATA_DIR,
    WEED_ANNOT_DIR,
    WEED_BASE_MODEL,
    WEED_BATCH_SIZE,
    WEED_BEST_WEIGHTS,
    WEED_CLASS_MAP,
    WEED_CLASS_NAMES,
    WEED_CONF_THRESHOLD,
    WEED_DATA_DIR,
    WEED_EPOCHS,
    WEED_IMG_SIZE,
    WEED_IMAGES_DIR,
    WEED_LABELS_DIR,
    WEED_PATIENCE,
    WEED_PROJECT_NAME,
    WEED_RUN_NAME,
    WEED_TRAIN_RATIO,
    WEED_VAL_RATIO,
    WEED_YOLO_DATASET,
    WEED_ZIP_PATH,
)


# ═══════════════════════════════════════════════════════════════════════════
# 0. ZIP CIKART
# ═══════════════════════════════════════════════════════════════════════════

def extract_weed_dataset():
    print("\n" + "=" * 60)
    print("  [0/4] WEED DATASET ZIP'TEN CIKARTILIYOR")
    print("=" * 60)

    if not WEED_ZIP_PATH.exists():
        raise FileNotFoundError(
            f"archive.zip bulunamadi: {WEED_ZIP_PATH}\n"
            "Lutfen Desktop'a koyun."
        )

    # Zaten cikartildiysa atla
    if WEED_IMAGES_DIR.exists() and any(WEED_IMAGES_DIR.glob("*.jpg")):
        count = len(list(WEED_IMAGES_DIR.glob("*.jpg")))
        print(f"  Zaten cikartilmis ({count} goruntu), atlaniyor.")
        return

    WEED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Kaynak : {WEED_ZIP_PATH}")
    print(f"  Hedef  : {WEED_DATA_DIR}")

    with zipfile.ZipFile(WEED_ZIP_PATH, "r") as z:
        for member in z.namelist():
            fname = member.rsplit("/", 1)[-1]

            # macOS metadata dosyalarini atla
            if fname.startswith("._") or "__MACOSX" in member or not fname:
                continue

            if member.endswith(".jpg") or member.endswith(".JPG"):
                dest = WEED_IMAGES_DIR / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            elif member.endswith(".xml"):
                dest = WEED_ANNOT_DIR / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    jpg_count = len(list(WEED_IMAGES_DIR.glob("*.jpg")))
    xml_count = len(list(WEED_ANNOT_DIR.glob("*.xml")))
    print(f"  Goruntu: {jpg_count}  |  XML: {xml_count}")


# ═══════════════════════════════════════════════════════════════════════════
# 0b. AGRI_DATA BIRLESTIR
# ═══════════════════════════════════════════════════════════════════════════

def merge_agri_dataset():
    print("\n" + "=" * 60)
    print("  [0b] AGRI_DATA BIRLESTIRME (class 1=weed → class 0)")
    print("=" * 60)

    if not AGRI_DATA_DIR.exists():
        print(f"  agri_data klasoru bulunamadi: {AGRI_DATA_DIR}")
        print("  Atlaniyor.")
        return

    WEED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    WEED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    txt_files  = list(AGRI_DATA_DIR.glob("*.txt"))
    copied     = 0
    skipped    = 0
    total_ann  = 0

    for txt_path in txt_files:
        # Sadece class 1 (weed) satirlarini al
        weed_lines = []
        try:
            for line in txt_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts[0] == "1":           # class 1 = weed
                    parts[0] = "0"            # remap → class 0
                    weed_lines.append(" ".join(parts))
        except Exception as e:
            print(f"  HATA {txt_path.name}: {e}")
            skipped += 1
            continue

        if not weed_lines:                    # bu gorselde hic weed yok
            skipped += 1
            continue

        # Goruntu dosyasini bul (.jpeg veya .jpg)
        img_src = None
        for ext in (".jpeg", ".jpg", ".JPG", ".JPEG"):
            candidate = AGRI_DATA_DIR / (txt_path.stem + ext)
            if candidate.exists():
                img_src = candidate
                break

        if img_src is None:
            skipped += 1
            continue

        # Goruntu kopyala (uzantıyı .jpg yap — tutarlilik icin)
        img_dst = WEED_IMAGES_DIR / (txt_path.stem + ".jpg")
        shutil.copy2(img_src, img_dst)

        # Etiketi yaz
        lbl_dst = WEED_LABELS_DIR / (txt_path.stem + ".txt")
        lbl_dst.write_text("\n".join(weed_lines), encoding="utf-8")

        total_ann += len(weed_lines)
        copied    += 1

    print(f"  Eklenen goruntu  : {copied}")
    print(f"  Atlanan (weed yok): {skipped}")
    print(f"  Eklenen annotation: {total_ann}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. XML → YOLO
# ═══════════════════════════════════════════════════════════════════════════

def convert_xml_to_yolo():
    print("\n" + "=" * 60)
    print("  [1/4] XML → YOLO DONUSTURME")
    print("=" * 60)

    WEED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    xml_files  = list(WEED_ANNOT_DIR.glob("*.xml"))
    converted  = 0
    skipped    = 0
    empty      = 0
    class_dist = {c: 0 for c in WEED_CLASS_NAMES}

    for xml_path in xml_files:
        try:
            root   = ET.parse(xml_path).getroot()
            size   = root.find("size")
            img_w  = int(size.find("width").text)
            img_h  = int(size.find("height").text)

            lines = []
            for obj in root.findall("object"):
                cls = obj.find("name").text.strip().lower()
                if cls not in WEED_CLASS_MAP:
                    continue

                bb   = obj.find("bndbox")
                xmin = int(float(bb.find("xmin").text))
                ymin = int(float(bb.find("ymin").text))
                xmax = int(float(bb.find("xmax").text))
                ymax = int(float(bb.find("ymax").text))

                xc = (xmin + xmax) / 2 / img_w
                yc = (ymin + ymax) / 2 / img_h
                bw = (xmax - xmin) / img_w
                bh = (ymax - ymin) / img_h

                lines.append(
                    f"{WEED_CLASS_MAP[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )
                class_dist[cls] += 1

            txt_path = WEED_LABELS_DIR / (xml_path.stem + ".txt")
            txt_path.write_text("\n".join(lines), encoding="utf-8")

            if lines:
                converted += 1
            else:
                empty += 1

        except Exception as e:
            print(f"  HATA {xml_path.name}: {e}")
            skipped += 1

    print(f"  Donusturulen : {converted}")
    print(f"  Bos (no-obj) : {empty}")
    print(f"  Hata         : {skipped}")
    print(f"  Sinif dagilimi:")
    for cls, cnt in class_dist.items():
        print(f"    {cls:<8}: {cnt}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def split_weed_dataset():
    print("\n" + "=" * 60)
    print(f"  [2/4] VERI BOLME ({int(WEED_TRAIN_RATIO*100)}/{int(WEED_VAL_RATIO*100)}/15)")
    print("=" * 60)

    for split in ("train", "val", "test"):
        (WEED_YOLO_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (WEED_YOLO_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_imgs = [
        f for f in WEED_IMAGES_DIR.glob("*.jpg")
        if (WEED_LABELS_DIR / (f.stem + ".txt")).exists()
    ]

    random.seed(42)
    random.shuffle(all_imgs)

    n       = len(all_imgs)
    n_train = int(n * WEED_TRAIN_RATIO)
    n_val   = int(n * WEED_VAL_RATIO)

    splits = {
        "train": all_imgs[:n_train],
        "val":   all_imgs[n_train : n_train + n_val],
        "test":  all_imgs[n_train + n_val :],
    }

    for split_name, files in splits.items():
        for img in files:
            shutil.copy2(img, WEED_YOLO_DATASET / "images" / split_name / img.name)
            lbl = WEED_LABELS_DIR / (img.stem + ".txt")
            dest_lbl = WEED_YOLO_DATASET / "labels" / split_name / lbl.name
            if lbl.exists():
                shutil.copy2(lbl, dest_lbl)
            else:
                dest_lbl.write_text("")
        print(f"  {split_name.upper():6}: {len(files)} goruntu")


# ═══════════════════════════════════════════════════════════════════════════
# 3. data.yaml
# ═══════════════════════════════════════════════════════════════════════════

def create_weed_yaml():
    print("\n" + "=" * 60)
    print("  [3/4] data.yaml OLUSTURMA")
    print("=" * 60)

    cfg = {
        "path":  str(WEED_YOLO_DATASET),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(WEED_CLASS_NAMES),
        "names": WEED_CLASS_NAMES,
    }

    yaml_path = WEED_YOLO_DATASET / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"  Siniflar: {WEED_CLASS_NAMES}")
    print(f"  Kayit   : {yaml_path}")
    return yaml_path


# ═══════════════════════════════════════════════════════════════════════════
# 4. EGITIM
# ═══════════════════════════════════════════════════════════════════════════

def train_weed_model():
    print("\n" + "=" * 60)
    print("  [4/4] YOLOV8 WEED MODEL EGITIMI")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"  Device : {gpu_name}")
    print(f"  Epochs : {WEED_EPOCHS}  |  Batch: {WEED_BATCH_SIZE}  |  ImgSize: {WEED_IMG_SIZE}")

    model     = YOLO(WEED_BASE_MODEL)
    yaml_path = WEED_YOLO_DATASET / "data.yaml"

    results = model.train(
        data=str(yaml_path),
        epochs=WEED_EPOCHS,
        imgsz=WEED_IMG_SIZE,
        batch=WEED_BATCH_SIZE,
        patience=WEED_PATIENCE,
        device=device,
        project=WEED_PROJECT_NAME,
        name=WEED_RUN_NAME,
        save=True,
        plots=True,
        conf=WEED_CONF_THRESHOLD,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n  Egitim tamamlandi!")
    print(f"  En iyi model: {best}")
    print(f"\n  Analiz icin:")
    print(f"  python weed_analyze.py --image <goruntu.jpg>")
    return best


# ═══════════════════════════════════════════════════════════════════════════
# ANA AKIS
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true",
                        help="ZIP zaten cikartildiysa atla")
    parser.add_argument("--skip-merge", action="store_true",
                        help="agri_data zaten birlestirildiyse atla")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  GOREV 4 — WEED DETECTION PIPELINE")
    print("=" * 60)

    if not args.skip_extract:
        extract_weed_dataset()

    if not args.skip_merge:
        merge_agri_dataset()

    convert_xml_to_yolo()
    split_weed_dataset()
    create_weed_yaml()
    train_weed_model()


if __name__ == "__main__":
    main()
