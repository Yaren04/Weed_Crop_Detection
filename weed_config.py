from pathlib import Path

# ── Proje Koku ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()

# ── Ham Veri ─────────────────────────────────────────────────────────────────
WEED_ZIP_PATH   = Path.home() / "Desktop" / "archive.zip"

# agri_data ek dataseti (Kaggle: crop and weed detection with bounding boxes)
# Class 0 = crop (susam)  → atilir
# Class 1 = weed          → class 0'a remap edilir
AGRI_DATA_DIR   = Path.home() / "Desktop" / "agri_data" / "data"

WEED_DATA_DIR   = ROOT / "weed_data"
WEED_IMAGES_DIR = WEED_DATA_DIR / "raw_images"    # zip'ten cikarilan gorseller
WEED_ANNOT_DIR  = WEED_DATA_DIR / "annotations"   # zip'ten cikarilan XML'ler
WEED_LABELS_DIR = WEED_DATA_DIR / "labels"        # XML → YOLO donusum ciktisi

# ── YOLO Dataset ─────────────────────────────────────────────────────────────
WEED_YOLO_DATASET = ROOT / "weed_yolo_dataset"
WEED_RESULTS_DIR  = ROOT / "results" / "weed"
WEED_MODEL_DIR    = ROOT / "weed_yolo"

# ── Siniflar ─────────────────────────────────────────────────────────────────
# Sadece "weed" sinifi: crop ornekleri cok az (411 vs 7442) ve class
# imbalance mAP'i asagi cekiyordu. Tek sinif → daha net sinir, daha yuksek mAP.
WEED_CLASS_NAMES = ["weed"]
WEED_CLASS_MAP   = {name: idx for idx, name in enumerate(WEED_CLASS_NAMES)}

WEED_CLASS_COLORS = {
    "weed": "#E74C3C",   # Kirmizi
}

# ── Veri Bolme ───────────────────────────────────────────────────────────────
WEED_TRAIN_RATIO = 0.70
WEED_VAL_RATIO   = 0.15
# test = kalan 0.15

# ── Model & Egitim ────────────────────────────────────────────────────────────
WEED_BASE_MODEL     = "yolov8m.pt"
WEED_EPOCHS         = 120
WEED_IMG_SIZE       = 640
WEED_BATCH_SIZE     = 8
WEED_PATIENCE       = 25
WEED_CONF_THRESHOLD = 0.2    # Weed icin biraz daha dusuk (kucuk otlari kacirma)
WEED_PROJECT_NAME   = "weed_yolo"
WEED_RUN_NAME       = "run1"

# ── Cikti Dosyasi ─────────────────────────────────────────────────────────────
WEED_BEST_WEIGHTS = WEED_MODEL_DIR / WEED_RUN_NAME / "weights" / "best.pt"
