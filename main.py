"""
Kullanim:
    conda activate crop_detection
    python main.py               # tam pipeline (zip'ten baslar)
    python main.py --skip-train  # egitim zaten bittiyse atla
    python main.py --skip-extract  # zip zaten cikartildiysa atla
"""

import argparse
import sys
from pathlib import Path

import pipeline


def find_best_weights():
    """runs/ altindaki en son egitilen modeli bul."""
    root = Path(__file__).parent
    candidates = list(root.rglob("weights/best.pt"))
    if not candidates:
        candidates = list(root.rglob("weights/last.pt"))
    if not candidates:
        return None
    # En son degistirilen dosyayi sec
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args():
    p = argparse.ArgumentParser(description="Vegetable Crops Tespit Pipeline")
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Mevcut best.pt kullan, yeniden egitme",
    )
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="ZIP zaten cikartildi, extract adimini atla",
    )
    return p.parse_args()


def step(label, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"\n  HATA [{label}]: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  VEGETABLE CROPS ERKEN BUYUME — PIPELINE BASLIYOR")
    print(f"  Siniflar: {', '.join(__import__('config').CLASS_NAMES)}")
    print("=" * 60)

    # 0. ZIP cikart
    if not args.skip_extract:
        step("ZIP Extract", pipeline.extract_dataset)
    else:
        print("\n  [0/10] ZIP extract atlandi (--skip-extract)")

    # 1-3. Hazirlik
    step("Annotation→YOLO", pipeline.convert_to_yolo)
    step("Split",           pipeline.split_dataset)
    step("data.yaml",       pipeline.create_data_yaml)

    # 4. Egitim
    if args.skip_train:
        weights = find_best_weights()
        if weights is None:
            print("\n  HATA: --skip-train verildi ama hicbir best.pt/last.pt bulunamadi.")
            print("  Once 'python main.py' ile egitimi tamamlayin.")
            sys.exit(1)
        from ultralytics import YOLO
        model = YOLO(str(weights))
        print(f"\n  Model yuklendi (egitim atlandi): {weights}")
    else:
        model, _ = step("Egitim", pipeline.train_model)

    # 5-10. Analiz + Rapor
    predictions         = step("Tahmin",  pipeline.run_predictions,       model)
    df_det, df_img, cls = step("Analiz",  pipeline.analyze_predictions,   predictions)
    step("Gorsel",  pipeline.create_visualizations,  df_det, df_img, cls)
    step("CSV",     pipeline.create_csv_reports,     df_det, df_img, cls)
    step("Metrik",  pipeline.evaluate_model,         model)
    step("Rapor",   pipeline.final_summary,          cls)

    print("\n" + "=" * 60)
    print("  PIPELINE TAMAMLANDI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
