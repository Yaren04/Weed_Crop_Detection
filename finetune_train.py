"""
Frozen backbone fine-tune.

Strateji:
  - Backbone (ilk 10 katman) dondurulur  → ogrenilenler korunur
  - Sadece detection head guncellenir    → sinif karisikligi azalir
  - LR cok dusuk tutulur                → salınim olmaz

Kullanim:
    Ctrl+C ile mevcut egitimi durdur, sonra:
    conda activate crop_detection
    python finetune_train.py
"""

from pathlib import Path
import torch
from ultralytics import YOLO
from crop.config import IMG_SIZE, BATCH_SIZE, YOLO_DATASET, CONF_THRESHOLD


def find_best_weights() -> Path:
    root = Path(__file__).parent
    # En son kaydedilen best.pt'yi bul (finetune > run1-2 > run1 sirasi ile)
    candidates = list(root.rglob("weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("best.pt bulunamadi.")
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"  Baslangic modeli: {best}")
    return best


def main():
    print(f"\n{'='*60}")
    print("  FROZEN BACKBONE FINE-TUNE")
    print("  Backbone donduruldu → sadece detection head ogrenir")
    print(f"{'='*60}\n")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")

    model = YOLO(str(find_best_weights()))

    results = model.train(
        data=str(YOLO_DATASET / "data.yaml"),
        epochs=60,
        imgsz=IMG_SIZE,
        batch=8,                # kucuk batch → daha az gurultu, daha stabil
        patience=15,
        device=device,
        project="crops_yolo",
        name="finetune",

        # ── Backbone dondurma ────────────────────────────────
        freeze=10,              # ilk 10 katmani dondur (backbone)
        # ────────────────────────────────────────────────────

        # ── Cok dusuk LR — salınim olmaz ────────────────────
        lr0=0.00005,            # 0.0001'in yarisi
        lrf=0.1,                # son LR = 0.000005
        warmup_epochs=0,        # warmup yok, zaten egitilmis
        # ────────────────────────────────────────────────────

        # ── Veri artirma azalt — ogrenilenler bozulmasin ────
        hsv_h=0.01,             # renk kaymasini azalt (default 0.015)
        hsv_s=0.5,
        flipud=0.0,             # dikey cevrime kapali
        mosaic=0.5,             # mozaik artirmayi azalt (default 1.0)
        # ────────────────────────────────────────────────────

        save=True,
        plots=True,
        conf=CONF_THRESHOLD,
        resume=False,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  Fine-tune tamamlandi!")
    print(f"  En iyi model: {best}")
    print(f"\n  Simdi analiz:")
    print(f"  python crop/analyze.py --image <goruntu.jpg>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
