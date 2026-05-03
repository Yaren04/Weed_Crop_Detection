from pathlib import Path

# ── Proje Kökü ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()

# ── Ham Veri (zip + cikartilmis klasor) ─────────────────────────────────────
ZIP_PATH     = Path.home() / "Desktop" / "DB.zip"   # indirilen zip
DATA_DIR     = ROOT / "data"
DATASET_DIR  = DATA_DIR / "dataset"        # zip icerisindeki duz klasor
LABELS_DIR   = DATA_DIR / "labels"         # XML/JSON → YOLO donusum ciktisi

# dataset icindeki hazir split listleri (zip'ten cikartilacak)
YOLO_TRAIN_TXT = DATA_DIR / "yolo-train.txt"
YOLO_VAL_TXT   = DATA_DIR / "yolo-val.txt"

# Test split: yolo-val.txt'nin yarisini test olarak ayiriyoruz
# (orijinal dataset'te test listesi yok)
TEST_SPLIT_RATIO = 0.50   # val dosyasinin %50'si → test

# ── YOLO Dataset Ciktisi ─────────────────────────────────────────────────────
YOLO_DATASET = ROOT / "yolo_dataset"
RESULTS_DIR  = ROOT / "results"
MODEL_DIR    = ROOT / "crops_yolo"

# ── Siniflar (6 sinif) ───────────────────────────────────────────────────────
CLASS_NAMES = ["maize", "bean", "leek", "stem_maize", "stem_bean", "stem_leek"]
CLASS_MAP   = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ── Etiket Hiyerarsisi ───────────────────────────────────────────────────────
# Birincil etiket: hepsi "Yeni Ekin/Fide"
# Ikincil etiket:  tespit edilen tur (maize/bean/leek) veya govde bilgisi

# Model sinifi → (birincil_etiket, ikincil_tur, govde_mi?)
CLASS_HIERARCHY = {
    "maize":      ("Fide",  "Misir",   False),
    "bean":       ("Fide",  "Fasulye", False),
    "leek":       ("Fide",  "Pirasa",  False),
    "stem_maize": ("Govde", "Misir",   True),
    "stem_bean":  ("Govde", "Fasulye", True),
    "stem_leek":  ("Govde", "Pirasa",  True),
}

# Tur bazli renkler (ikincil etiket rengi)
TYPE_COLORS = {
    "Misir":   "#F4D03F",
    "Fasulye": "#2ECC71",
    "Pirasa":  "#3498DB",
}

# Birincil etiket renkleri
PRIMARY_COLORS = {
    "Fide":  "#E74C3C",   # Kirmizi — yeni ekin
    "Govde": "#95A5A6",   # Gri    — govde (ikincil bilgi)
}

# Geriye donuk uyumluluk
CLASS_COLORS = {
    "maize":      TYPE_COLORS["Misir"],
    "bean":       TYPE_COLORS["Fasulye"],
    "leek":       TYPE_COLORS["Pirasa"],
    "stem_maize": "#E67E22",
    "stem_bean":  "#27AE60",
    "stem_leek":  "#2980B9",
}

# ── Model & Egitim ───────────────────────────────────────────────────────────
BASE_MODEL     = "yolov8m.pt"
EPOCHS         = 150    # 50→150: egri hala yukseliyordu, devam ediyoruz
IMG_SIZE       = 640
BATCH_SIZE     = 16
PATIENCE       = 20     # 10→20: erken durmasin, daha sabırli beklesin
CONF_THRESHOLD = 0.5
PROJECT_NAME   = "crops_yolo"
RUN_NAME       = "run2"  # run1'den devam — ayri klasore kaydeder

# ── Cikti Dosyalari ──────────────────────────────────────────────────────────
BEST_WEIGHTS = MODEL_DIR / RUN_NAME / "weights" / "best.pt"
