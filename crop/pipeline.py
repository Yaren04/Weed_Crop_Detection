"""
Tum pipeline adimlari. main.py tarafindan cagrilir.

Adimlar:
  0. extract_dataset     — DB.zip'i data/ altina cikart
  1. convert_to_yolo     — XML (+ JSON fallback) → YOLO txt
  2. split_dataset       — yolo-train/val.txt kullanarak klasorlere dagit
  3. create_data_yaml    — yolo_dataset/data.yaml olustur
  4. train_model         — YOLOv8m egit
  5. run_predictions     — test seti uzerinde inference
  6. analyze_predictions — sayim + istatistik
  7. create_visualizations
  8. create_csv_reports
  9. evaluate_model      — mAP metrikleri
 10. final_summary       — FINAL_REPORT.txt
"""

import csv
import json
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from ultralytics import YOLO

from crop.config import (
    BATCH_SIZE,
    BASE_MODEL,
    BEST_WEIGHTS,
    CLASS_COLORS,
    CLASS_MAP,
    CLASS_NAMES,
    CONF_THRESHOLD,
    DATA_DIR,
    DATASET_DIR,
    EPOCHS,
    IMG_SIZE,
    LABELS_DIR,
    MODEL_DIR,
    PATIENCE,
    PROJECT_NAME,
    RESULTS_DIR,
    RUN_NAME,
    TEST_SPLIT_RATIO,
    YOLO_DATASET,
    YOLO_TRAIN_TXT,
    YOLO_VAL_TXT,
    ZIP_PATH,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 0. ZIP'TEN CIKART
# ═══════════════════════════════════════════════════════════════════════════════

def extract_dataset():
    print("\n" + "=" * 60)
    print("  [0/10] DATASET ZIP'TEN CIKARTILIYOR")
    print("=" * 60)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"DB.zip bulunamadi: {ZIP_PATH}\n"
            "Lutfen dosyayi Desktop'a indirin."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Zaten cikartildiysa atla
    if DATASET_DIR.exists() and any(DATASET_DIR.glob("*.jpg")):
        jpg_count = len(list(DATASET_DIR.glob("*.jpg")))
        print(f"  Zaten cikartilmis ({jpg_count} goruntu bulundu), atlaniyor.")
        return

    print(f"  Kaynak : {ZIP_PATH}")
    print(f"  Hedef  : {DATA_DIR}")
    print("  Cikartiliyor... (bu 1-2 dakika surebilir)")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = z.namelist()

        for member in members:
            # DB_Mendeley/ on ekini sil → data/ altina yerlestir
            # ornekler:
            #   DB_Mendeley/dataset/foo.jpg  → data/dataset/foo.jpg
            #   DB_Mendeley/yolo-train.txt   → data/yolo-train.txt
            relative = member.removeprefix("DB_Mendeley/")
            fname = relative.rsplit("/", 1)[-1]
            # macOS metadata dosyalarini atla (._* ve __MACOSX)
            if not relative or relative == "/" or fname.startswith("._") or "__MACOSX" in relative:
                continue

            dest = DATA_DIR / relative

            info = z.getinfo(member)
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    jpg_count = len(list(DATASET_DIR.glob("*.jpg")))
    xml_count = len(list(DATASET_DIR.glob("*.xml")))
    print(f"  Tamamlandi: {jpg_count} goruntu, {xml_count} XML")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. XML / JSON → YOLO TXT
# ═══════════════════════════════════════════════════════════════════════════════

def _xml_to_yolo_lines(xml_path: Path) -> list[str]:
    """Bir XML dosyasini YOLO formati satirlarina donustur."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text.strip().lower()
        if cls not in CLASS_MAP:
            continue
        bb = obj.find("bndbox")
        xmin = int(bb.find("xmin").text)
        ymin = int(bb.find("ymin").text)
        xmax = int(bb.find("xmax").text)
        ymax = int(bb.find("ymax").text)
        xc = (xmin + xmax) / 2 / w
        yc = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{CLASS_MAP[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _json_to_yolo_lines(json_path: Path, img_path: Path) -> list[str]:
    """JSON formatini YOLO formati satirlarina donustur (XML yoksa fallback)."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for obj in data.get("objects", []):
        cls = obj.get("label", "").strip().lower()
        if cls not in CLASS_MAP:
            continue
        box = obj.get("box", {})
        xmin = box.get("x_min", 0)
        ymin = box.get("y_min", 0)
        xmax = box.get("x_max", 0)
        ymax = box.get("y_max", 0)
        xc = (xmin + xmax) / 2 / w
        yc = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{CLASS_MAP[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def convert_to_yolo():
    print("\n" + "=" * 60)
    print("  [1/10] ANNOTATION → YOLO DONUSTURME")
    print("=" * 60)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    all_images = [
        f for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG")
        for f in DATASET_DIR.glob(ext)
        if not f.name.startswith("._")   # macOS metadata dosyalarini atla
    ]

    converted_xml  = 0
    converted_json = 0
    skipped        = 0
    unknown_classes: set[str] = set()

    for img_path in all_images:
        stem       = img_path.stem
        xml_path   = DATASET_DIR / (stem + ".xml")
        json_path  = DATASET_DIR / (stem + ".json")
        txt_path   = LABELS_DIR  / (stem + ".txt")

        lines = []

        if xml_path.exists():
            try:
                lines = _xml_to_yolo_lines(xml_path)
                # Bilinmeyen siniflari raporla (sessizce)
                root = ET.parse(xml_path).getroot()
                for obj in root.findall("object"):
                    cls = obj.find("name").text.strip().lower()
                    if cls not in CLASS_MAP:
                        unknown_classes.add(cls)
                converted_xml += 1
            except Exception as e:
                print(f"  UYARI XML {img_path.name}: {e}")
                skipped += 1
                continue

        elif json_path.exists():
            try:
                lines = _json_to_yolo_lines(json_path, img_path)
                converted_json += 1
            except Exception as e:
                print(f"  UYARI JSON {img_path.name}: {e}")
                skipped += 1
                continue
        else:
            # Annotation yok → bos label dosyasi (arka plan / no-object)
            skipped += 1

        txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  XML ile donusturulen  : {converted_xml}")
    print(f"  JSON ile donusturulen : {converted_json}")
    print(f"  Annotation yok (atla) : {skipped}")
    if unknown_classes:
        print(f"  Bilinmeyen siniflar   : {unknown_classes} → atland1")
    print(f"  Cikti klasoru         : {LABELS_DIR}")
    return LABELS_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT — yolo-train.txt / yolo-val.txt kullan
# ═══════════════════════════════════════════════════════════════════════════════

def split_dataset():
    print("\n" + "=" * 60)
    print("  [2/10] VERI DAGILIMI (resmi train/val + test ayirimi)")
    print("=" * 60)

    # Hedef klasorleri olustur
    for split in ("train", "val", "test"):
        (YOLO_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _read_list(txt_path: Path) -> list[str]:
        return [
            ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    train_names = _read_list(YOLO_TRAIN_TXT)

    # Val listesini test split olarak bolelim
    val_names_all = _read_list(YOLO_VAL_TXT)
    random.seed(42)
    random.shuffle(val_names_all)
    n_test = int(len(val_names_all) * TEST_SPLIT_RATIO)
    test_names = val_names_all[:n_test]
    val_names  = val_names_all[n_test:]

    splits = {
        "train": train_names,
        "val":   val_names,
        "test":  test_names,
    }

    missing_imgs = 0
    for split_name, file_names in splits.items():
        copied = 0
        for fname in file_names:
            img_src = DATASET_DIR / fname
            if not img_src.exists():
                missing_imgs += 1
                continue
            stem = Path(fname).stem
            lbl_src = LABELS_DIR / (stem + ".txt")

            shutil.copy2(img_src, YOLO_DATASET / "images" / split_name / fname)

            if lbl_src.exists():
                shutil.copy2(
                    lbl_src,
                    YOLO_DATASET / "labels" / split_name / (stem + ".txt"),
                )
            else:
                # Bos label dosyasi olustur (no-object goruntu)
                (YOLO_DATASET / "labels" / split_name / (stem + ".txt")).write_text("")
            copied += 1

        print(f"  {split_name.upper():6}: {copied} goruntu")

    if missing_imgs:
        print(f"  UYARI: {missing_imgs} goruntu DATASET_DIR'de bulunamadi")

    print(f"  Kaynak : yolo-train.txt ({len(train_names)}) + yolo-val.txt ({len(val_names_all)})")
    print(f"  Test   : val'in %{int(TEST_SPLIT_RATIO*100)}'i ({len(test_names)} goruntu)")
    return YOLO_DATASET


# ═══════════════════════════════════════════════════════════════════════════════
# 3. data.yaml
# ═══════════════════════════════════════════════════════════════════════════════

def create_data_yaml():
    print("\n" + "=" * 60)
    print("  [3/10] data.yaml OLUSTURMA")
    print("=" * 60)

    cfg = {
        "path":  str(YOLO_DATASET),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = YOLO_DATASET / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"  Siniflar ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    print(f"  Kaydedildi: {yaml_path}")
    return yaml_path


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL EGITIMI
# ═══════════════════════════════════════════════════════════════════════════════

def train_model():
    print("\n" + "=" * 60)
    print("  [4/10] YOLOV8 MODEL EGITIMI")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")
    print(f"  Epochs: {EPOCHS}  |  ImgSize: {IMG_SIZE}  |  Batch: {BATCH_SIZE}")

    model = YOLO(BASE_MODEL)
    yaml_path = YOLO_DATASET / "data.yaml"

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        project=PROJECT_NAME,
        name=RUN_NAME,
        save=True,
        plots=True,
        conf=CONF_THRESHOLD,
    )

    # YOLO gercek kayit klasorunu results.save_dir'den al
    # (runs/detect/ oneki ve run1-2 gibi otomatik isimler icin guvenli yol)
    actual_best = Path(results.save_dir) / "weights" / "best.pt"
    if not actual_best.exists():
        # last.pt'ye dus
        actual_best = Path(results.save_dir) / "weights" / "last.pt"
    print(f"  En iyi agirliklar: {actual_best}")
    return YOLO(str(actual_best)), results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TAHMİN
# ═══════════════════════════════════════════════════════════════════════════════

def run_predictions(model):
    print("\n" + "=" * 60)
    print("  [5/10] TEST TAHMINI")
    print("=" * 60)

    test_dir = YOLO_DATASET / "images" / "test"
    device   = 0 if torch.cuda.is_available() else "cpu"

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
        image_files.extend(sorted(test_dir.glob(ext)))

    predictions = []
    for img in image_files:
        pred = model.predict(
            source=str(img), conf=CONF_THRESHOLD, device=device, verbose=False
        )
        predictions.append({"image": img.name, "prediction": pred[0]})

    print(f"  {len(predictions)} goruntu islendi")
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANALİZ
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_predictions(predictions):
    print("\n" + "=" * 60)
    print("  [6/10] BITKI SAYIMI VE ANALIZ")
    print("=" * 60)

    class_totals  = defaultdict(int)
    all_detections = []
    image_summaries = []

    for item in predictions:
        img_name = item["image"]
        pred     = item["prediction"]
        boxes    = pred.boxes
        count    = 0

        if boxes is not None:
            for i in range(len(boxes)):
                cid  = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i]
                cls  = CLASS_NAMES[cid]
                class_totals[cls] += 1
                count += 1
                all_detections.append({
                    "image":      img_name,
                    "class":      cls,
                    "confidence": conf,
                    "x1": float(xyxy[0]), "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]), "y2": float(xyxy[3]),
                })
        image_summaries.append({"image": img_name, "total_detections": count})

    total = sum(class_totals.values())
    print(f"\n  {'SINIF':<15} {'SAYI':>6}  {'ORAN':>7}")
    print(f"  {'-'*32}")
    for cls in CLASS_NAMES:
        cnt = class_totals[cls]
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 4)
        print(f"  {cls.upper():<15} {cnt:>6}  ({pct:5.1f}%) {bar}")
    print(f"  {'-'*32}")
    print(f"  {'TOPLAM':<15} {total:>6}")

    return (
        pd.DataFrame(all_detections),
        pd.DataFrame(image_summaries),
        dict(class_totals),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════════

def create_visualizations(df_det, df_img, class_counts):
    print("\n" + "=" * 60)
    print("  [7/10] GORSELLESTIRME")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    counts = [class_counts.get(c, 0) for c in CLASS_NAMES]
    total  = sum(counts)
    colors = list(CLASS_COLORS.values())

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Vegetable Crops — Tespit Analiz Raporu", fontsize=14, fontweight="bold")

    # 1. Pie — tam dagilim
    ax1 = fig.add_subplot(2, 3, 1)
    wedges, texts, autotexts = ax1.pie(
        counts, labels=CLASS_NAMES, autopct="%1.1f%%", colors=colors, startangle=90
    )
    ax1.set_title("Sinif Dagilimi", fontweight="bold")

    # 2. Bar — sayim
    ax2 = fig.add_subplot(2, 3, 2)
    bars = ax2.bar(CLASS_NAMES, counts, color=colors, edgecolor="black")
    ax2.set_ylabel("Tespit Sayisi")
    ax2.set_title("Sinif Bazinda Tespit Sayisi", fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    for bar, cnt in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(cnt), ha="center", fontsize=9, fontweight="bold",
        )

    # 3. Confidence histogram
    ax3 = fig.add_subplot(2, 3, 3)
    if not df_det.empty:
        for cls, color in zip(CLASS_NAMES, colors):
            sub = df_det[df_det["class"] == cls]["confidence"]
            if not sub.empty:
                ax3.hist(sub, bins=15, alpha=0.6, color=color, label=cls)
        ax3.set_xlabel("Confidence")
        ax3.set_ylabel("Frekans")
        ax3.set_title("Sinif Bazinda Confidence Dagilimi", fontweight="bold")
        ax3.legend(fontsize=7)

    # 4. Boxplot — confidence per class
    ax4 = fig.add_subplot(2, 3, 4)
    if not df_det.empty:
        data_groups = [df_det[df_det["class"] == c]["confidence"].values for c in CLASS_NAMES]
        bp = ax4.boxplot(data_groups, labels=CLASS_NAMES, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax4.tick_params(axis="x", rotation=30)
        ax4.set_ylabel("Confidence")
        ax4.set_title("Sinif Bazinda Confidence", fontweight="bold")

    # 5. Top 10 goruntu
    ax5 = fig.add_subplot(2, 3, 5)
    if not df_img.empty:
        top = df_img.nlargest(10, "total_detections")
        ax5.barh(range(len(top)), top["total_detections"], color="#E74C3C", edgecolor="black")
        ax5.set_yticks(range(len(top)))
        ax5.set_yticklabels(
            [n.rsplit(".", 1)[0][:30] for n in top["image"]], fontsize=8
        )
        ax5.set_xlabel("Tespit Sayisi")
        ax5.set_title("En Cok Tespit Yapilan 10 Goruntu", fontweight="bold")

    # 6. Ozet metin
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    avg_conf = df_det["confidence"].mean() if not df_det.empty else 0
    plant_total = sum(class_counts.get(c, 0) for c in ["maize", "bean", "leek"])
    stem_total  = sum(class_counts.get(c, 0) for c in ["stem_maize", "stem_bean", "stem_leek"])
    lines = [
        "OZET",
        f"Toplam Tespit  : {total}",
        f"  Bitkiler     : {plant_total}",
        f"  Govdeler     : {stem_total}",
        f"Goruntu Sayisi : {len(df_img)}",
        f"Ort. Conf.     : {avg_conf:.3f}",
        "",
    ] + [
        f"{c:<12}: {class_counts.get(c,0):>5}" for c in CLASS_NAMES
    ]
    ax6.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax6.transAxes, fontsize=10,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    out = RESULTS_DIR / "analysis_report.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {out}")
    return RESULTS_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CSV RAPORLAR
# ═══════════════════════════════════════════════════════════════════════════════

def create_csv_reports(df_det, df_img, class_counts):
    print("\n" + "=" * 60)
    print("  [8/10] CSV RAPORLAR")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(class_counts.values())

    df_det.to_csv(RESULTS_DIR / "detections_detailed.csv",  index=False)
    df_img.to_csv(RESULTS_DIR / "images_summary.csv",       index=False)

    with open(RESULTS_DIR / "group_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sinif", "Kategori", "Sayi", "Yuzde"])
        for cls in CLASS_NAMES:
            cnt  = class_counts.get(cls, 0)
            cat  = "Govde" if cls.startswith("stem_") else "Bitki"
            pct  = f"{cnt/total*100:.2f}%" if total else "0.00%"
            w.writerow([cls, cat, cnt, pct])
        w.writerow(["TOPLAM", "-", total, "100.00%"])

    with open(RESULTS_DIR / "statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metrik", "Deger"])
        w.writerow(["Tarih", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        w.writerow(["Toplam Tespit", total])
        w.writerow(["Toplam Goruntu", len(df_img)])
        w.writerow(["Ort. Tespit/Goruntu", f"{total/max(len(df_img),1):.2f}"])
        if not df_det.empty:
            w.writerow(["Ort. Confidence", f"{df_det['confidence'].mean():.4f}"])
            w.writerow(["Min Confidence",  f"{df_det['confidence'].min():.4f}"])
            w.writerow(["Max Confidence",  f"{df_det['confidence'].max():.4f}"])

    for name in ("detections_detailed", "images_summary", "group_analysis", "statistics"):
        print(f"  {name}.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MODEL DEĞERLENDİRMESİ
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model):
    print("\n" + "=" * 60)
    print("  [9/10] MODEL DEGERLENDIRMESI")
    print("=" * 60)

    yaml_path = YOLO_DATASET / "data.yaml"
    metrics = model.val(data=str(yaml_path))
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5-0.95 : {metrics.box.map:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 10. FİNAL RAPOR
# ═══════════════════════════════════════════════════════════════════════════════

def final_summary(class_counts):
    print("\n" + "=" * 60)
    print("  [10/10] FINAL RAPOR")
    print("=" * 60)

    total       = sum(class_counts.values())
    plant_total = sum(class_counts.get(c, 0) for c in ["maize", "bean", "leek"])
    stem_total  = sum(class_counts.get(c, 0) for c in ["stem_maize", "stem_bean", "stem_leek"])

    lines = [
        "=" * 68,
        "  VEGETABLE CROPS ERKEN BUYUME — TESPIT RAPORU",
        f"  Model   : YOLOv8m  |  Sinif Sayisi: {len(CLASS_NAMES)}",
        f"  Tarih   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 68,
        "",
        "  TESPIT SAYILARI:",
        "-" * 68,
        "  [ BİTKİLER ]",
    ]
    for cls in ["maize", "bean", "leek"]:
        cnt = class_counts.get(cls, 0)
        pct = cnt / total * 100 if total else 0
        lines.append(f"    {cls.capitalize():<20}: {cnt:>6}  ({pct:.1f}%)")

    lines += ["", "  [ GOVDELER ]"]
    for cls in ["stem_maize", "stem_bean", "stem_leek"]:
        cnt = class_counts.get(cls, 0)
        pct = cnt / total * 100 if total else 0
        lines.append(f"    {cls.capitalize():<20}: {cnt:>6}  ({pct:.1f}%)")

    lines += [
        "-" * 68,
        f"  {'Bitkiler topiam':<22}: {plant_total:>6}",
        f"  {'Govdeler toplam':<22}: {stem_total:>6}",
        f"  {'GENEL TOPLAM':<22}: {total:>6}",
        "",
        "  CIKTI DOSYALARI:",
        f"  {RESULTS_DIR}",
        "    - analysis_report.png",
        "    - detections_detailed.csv",
        "    - images_summary.csv",
        "    - group_analysis.csv",
        "    - statistics.csv",
        f"  {BEST_WEIGHTS}",
        "=" * 68,
    ]

    report = "\n".join(lines)
    print(report)

    out = RESULTS_DIR / "FINAL_REPORT.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n  Kaydedildi: {out}")
