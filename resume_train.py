"""
Mevcut egitimden devam et (run1 → run2).
50 epoch tekrar egitmek zorunda kalmadan kaldigi yerden surer.

Kullanim:
    conda activate crop_detection
    python resume_train.py
"""

from pathlib import Path
import torch
from ultralytics import YOLO
from crop.config import (
    EPOCHS, IMG_SIZE, BATCH_SIZE, PATIENCE,
    PROJECT_NAME, CONF_THRESHOLD, YOLO_DATASET
)

def find_last_checkpoint() -> Path:
    """En son egitimin last.pt dosyasini bul."""
    root = Path(__file__).parent
    candidates = list(root.rglob("weights/last.pt"))
    if not candidates:
        raise FileNotFoundError("Hic egitim bulunamadi. Once 'python main.py' calistirin.")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def main():
    last_pt = find_last_checkpoint()
    print(f"\n{'='*60}")
    print(f"  EGITIM DEVAM EDİYOR")
    print(f"  Baslangic: {last_pt}")
    print(f"  Hedef    : {EPOCHS} epoch toplam")
    print(f"{'='*60}\n")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")

    model = YOLO(str(last_pt))

    results = model.train(
        data=str(YOLO_DATASET / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        project=PROJECT_NAME,
        name="run2",
        resume=True,      # ← kaldigi yerden devam
        save=True,
        plots=True,
        conf=CONF_THRESHOLD,
    )

    from pathlib import Path
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n  Tamamlandi!")
    print(f"  En iyi model: {best}")
    print(f"\n  Simdi analiz icin:")
    print(f"  python crop/analyze.py --image <goruntu.jpg>")

if __name__ == "__main__":
    main()
