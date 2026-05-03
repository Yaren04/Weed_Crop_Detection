"""
DB.zip Desktop'a indirildikten sonra calistir:
    conda activate crop_detection
    python check_dataset.py
"""

import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from crop.config import (
    CLASS_NAMES,
    DATA_DIR,
    DATASET_DIR,
    YOLO_TRAIN_TXT,
    YOLO_VAL_TXT,
    ZIP_PATH,
)

OK   = "  [OK]"
WARN = "  [!!]"


# ── 1. ZIP kontrolu ───────────────────────────────────────────────────────────
def check_zip():
    if ZIP_PATH.exists():
        size_mb = ZIP_PATH.stat().st_size / 1_048_576
        print(f"{OK} DB.zip bulundu ({size_mb:.1f} MB)  →  {ZIP_PATH}")
        return True
    print(f"{WARN} DB.zip bulunamadi: {ZIP_PATH}")
    print("       Lutfen dosyayi Desktop'a indirin.")
    return False


# ── 2. Cikartilmis dataset kontrolu ──────────────────────────────────────────
def check_extracted():
    if not DATASET_DIR.exists():
        print(f"{WARN} data/dataset/ klasoru yok — once 'python main.py' calistirin")
        return False

    jpgs = list(DATASET_DIR.glob("*.jpg")) + list(DATASET_DIR.glob("*.JPG"))
    xmls = list(DATASET_DIR.glob("*.xml"))
    jsns = list(DATASET_DIR.glob("*.json"))

    ok = bool(jpgs)
    print(f"{OK if ok else WARN} Goruntu  : {len(jpgs)}")
    print(f"{OK if xmls else WARN} XML      : {len(xmls)}")
    print(f"  [--] JSON     : {len(jsns)}")
    return ok


# ── 3. Split dosyalari kontrolu ───────────────────────────────────────────────
def check_split_files():
    all_ok = True
    for txt in (YOLO_TRAIN_TXT, YOLO_VAL_TXT):
        if txt.exists():
            lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
            print(f"{OK} {txt.name:<20}: {len(lines)} goruntu")
        else:
            print(f"{WARN} {txt.name} bulunamadi (main.py extract edince olusacak)")
            all_ok = False
    return all_ok


# ── 4. Sinif kontrolu ─────────────────────────────────────────────────────────
def check_classes(sample=10):
    if not DATASET_DIR.exists():
        return

    xml_files = list(DATASET_DIR.glob("*.xml"))[:sample]
    found = Counter()

    for xf in xml_files:
        try:
            root = ET.parse(xf).getroot()
            for obj in root.findall("object"):
                found[obj.find("name").text.strip().lower()] += 1
        except Exception:
            pass

    print(f"\n  Ornek XML siniflar (ilk {len(xml_files)} dosya):")
    for cls, cnt in found.most_common():
        tag = OK if cls in CLASS_NAMES else WARN + " TANIMLI DEGIL"
        print(f"    {tag}  '{cls}' ({cnt}x)")

    unknown = [c for c in found if c not in CLASS_NAMES]
    if unknown:
        print(f"\n  UYARI: Bilinmeyen siniflar config.py'de eksik: {unknown}")


# ── 5. ZIP icerisindeki ozet (cikartilmadan) ──────────────────────────────────
def check_zip_contents():
    if not ZIP_PATH.exists():
        return
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = z.namelist()
    exts = Counter()
    for m in members:
        if "dataset/" in m and "." in m.rsplit("/", 1)[-1]:
            exts[m.rsplit(".", 1)[-1].lower()] += 1
    print(f"\n  ZIP icindeki dataset dosyalari:")
    for ext, cnt in exts.most_common():
        print(f"    .{ext:<6}: {cnt}")


# ── Ana akis ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DATASET DOGRULAMA")
    print("=" * 60)

    zip_ok       = check_zip()
    print()
    extracted_ok = check_extracted()
    print()
    split_ok     = check_split_files()

    if zip_ok and not extracted_ok:
        check_zip_contents()

    check_classes()

    print("\n" + "=" * 60)
    if zip_ok and extracted_ok:
        print("  HAZIR — python main.py ile pipeline'i baslat.")
    elif zip_ok:
        print("  ZIP HAZIR — 'python main.py' calistir (otomatik extract eder).")
    else:
        print("  EKSIK — DB.zip'i Desktop'a indirin.")
    print("=" * 60)


if __name__ == "__main__":
    main()
