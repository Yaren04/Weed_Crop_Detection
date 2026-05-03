"""
Birlesik Tarla Analizi -- Crop + Weed Tespiti

Her iki model ayni goruntu uzerinde calistirilir:
  - Crop modeli : fide/govde tespiti + tur siniflandirmasi
  - Weed modeli : yabanci ot tespiti

Risk analizi:
  KIRMIZI (HIGH) : ot < 80 px uzakta
  TURUNCU (MED)  : ot 80-200 px uzakta
  YESIL   (LOW)  : ot > 200 px uzakta

Kullanim:
    conda activate crop_detection
    python field_combined_analyze.py --image tarla.jpg
    python field_combined_analyze.py --folder klasor/
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from crop.config import (
    CLASS_HIERARCHY,
    CLASS_NAMES,
    CONF_THRESHOLD,
    TYPE_COLORS,
)
from weed.config import WEED_CONF_THRESHOLD

# ── Sabitler ──────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).parent.resolve()
COMBINED_OUT_DIR = ROOT / "results" / "combined"

RISK_HIGH_PX = 80    # kirmizi esik (piksel)
RISK_MED_PX  = 200   # turuncu esik (piksel)

RISK_COLORS_BGR = {
    "HIGH": (0,   0,   220),   # kirmizi
    "MED":  (0,   140, 255),   # turuncu
    "LOW":  (0,   180, 0  ),   # yesil
}
WEED_BGR = (60, 60, 220)       # ot kutulari (kirmizi)


# ── Yardimcilar ───────────────────────────────────────────────────────────────

def hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


TYPE_BGR = {t: hex_to_bgr(c) for t, c in TYPE_COLORS.items()}


def put_label(img, text, x1, y1, color_bgr):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    y_top = max(y1 - th - 6, 0)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 4, y1), color_bgr, -1)
    cv2.putText(img, text, (x1 + 2, max(y1 - 4, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=8):
    """Kesikli cizgi -- yuksek riskli mahsulden ota."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    steps = int(dist / gap)
    for i in range(0, steps, 2):
        t0 = i / steps
        t1 = min((i + 1) / steps, 1.0)
        p0 = (int(x1 + t0 * (x2 - x1)), int(y1 + t0 * (y2 - y1)))
        p1 = (int(x1 + t1 * (x2 - x1)), int(y1 + t1 * (y2 - y1)))
        cv2.line(img, p0, p1, color, thickness)


# ── Model Bulma ───────────────────────────────────────────────────────────────

def find_crop_model() -> Path:
    candidates = [
        p for p in ROOT.rglob("weights/best.pt")
        if "weed" not in str(p).lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Crop modeli bulunamadi.\n"
            "Once 'python pipeline.py' ile egitim tamamlayin."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_weed_model() -> Path:
    candidates = list(ROOT.glob("weed_yolo/**/weights/best.pt"))
    if not candidates:
        candidates = [
            p for p in ROOT.rglob("weights/best.pt")
            if "weed" in str(p).lower()
        ]
    if not candidates:
        raise FileNotFoundError(
            "Weed modeli bulunamadi.\n"
            "Once 'python weed_pipeline.py' ile egitim tamamlayin."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── Tespit ────────────────────────────────────────────────────────────────────

def run_crop_detection(model, image_path: Path) -> pd.DataFrame:
    result = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        verbose=False,
    )[0]

    rows = []
    if result.boxes is None:
        return pd.DataFrame()

    img_h, img_w = result.orig_shape
    for i, box in enumerate(result.boxes):
        cid  = int(box.cls[0])
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "unknown"
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        primary, plant_type, is_stem = CLASS_HIERARCHY.get(
            name, ("Fide", "Bilinmeyen", False)
        )
        rows.append({
            "plant_id":   i + 1,
            "raw_class":  name,
            "primary":    primary,
            "type":       plant_type,
            "is_stem":    is_stem,
            "confidence": round(conf, 3),
            "center_x":   round((x1 + x2) / 2, 1),
            "center_y":   round((y1 + y2) / 2, 1),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
            "img_w": img_w, "img_h": img_h,
        })

    return pd.DataFrame(rows)


def run_weed_detection(model, image_path: Path) -> pd.DataFrame:
    result = model.predict(
        source=str(image_path),
        conf=WEED_CONF_THRESHOLD,
        verbose=False,
    )[0]

    rows = []
    if result.boxes is None:
        return pd.DataFrame()

    img_h, img_w = result.orig_shape
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append({
            "det_id":     i + 1,
            "confidence": round(conf, 3),
            "center_x":   round((x1 + x2) / 2, 1),
            "center_y":   round((y1 + y2) / 2, 1),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
            "img_w": img_w, "img_h": img_h,
        })

    return pd.DataFrame(rows)


# ── Risk Hesabi ───────────────────────────────────────────────────────────────

def calculate_risk(crops_df: pd.DataFrame, weeds_df: pd.DataFrame) -> pd.DataFrame:
    """Her mahsule en yakin otin mesafesini hesapla, risk seviyesi ata."""
    if crops_df.empty:
        return crops_df

    crops_df = crops_df.copy()

    if weeds_df.empty:
        crops_df["nearest_weed_px"] = np.inf
        crops_df["nearest_weed_id"] = -1
        crops_df["risk"]            = "LOW"
        return crops_df

    weed_coords = weeds_df[["center_x", "center_y"]].values  # (W, 2)
    weed_ids    = weeds_df["det_id"].values

    nearest_dist = []
    nearest_id   = []
    for _, row in crops_df.iterrows():
        cx, cy = row["center_x"], row["center_y"]
        dists  = np.hypot(weed_coords[:, 0] - cx, weed_coords[:, 1] - cy)
        idx    = int(np.argmin(dists))
        nearest_dist.append(round(float(dists[idx]), 1))
        nearest_id.append(int(weed_ids[idx]))

    crops_df["nearest_weed_px"] = nearest_dist
    crops_df["nearest_weed_id"] = nearest_id
    crops_df["risk"] = crops_df["nearest_weed_px"].apply(
        lambda d: "HIGH" if d < RISK_HIGH_PX
                  else ("MED" if d < RISK_MED_PX else "LOW")
    )
    return crops_df


# ── Gorseller ─────────────────────────────────────────────────────────────────

def draw_combined(image_path: Path, crops_df: pd.DataFrame,
                  weeds_df: pd.DataFrame, out_path: Path):
    """Crop kutulari (tur rengi) + weed kutulari (kirmizi) ayni kare."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    # Weed kutulari (ince, kirmizi)
    for _, r in weeds_df.iterrows():
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), WEED_BGR, 1)
        put_label(img, f"ot {r.confidence:.2f}", x1, y1, WEED_BGR)

    # Crop kutulari (tur renginde)
    for _, r in crops_df.iterrows():
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        color = TYPE_BGR.get(r["type"], (200, 200, 200))
        lw    = 1 if r["is_stem"] else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
        put_label(img, f"{r['type']} {r.confidence:.2f}", x1, y1, color)

    # Sol ust ozet
    cv2.putText(img, f"Mahsul: {len(crops_df)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"Ot    : {len(weeds_df)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WEED_BGR, 2)

    cv2.imwrite(str(out_path), img)
    print(f"  Birlesik gorsel : {out_path.name}")


def draw_risk(image_path: Path, crops_df: pd.DataFrame,
              weeds_df: pd.DataFrame, out_path: Path):
    """Crop kutulari risk renginde; HIGH risk icin ota kesikli cizgi."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    # Weed merkezleri -- yari saydam kirmizi daire
    if not weeds_df.empty:
        overlay = img.copy()
        for _, r in weeds_df.iterrows():
            cx, cy = int(r.center_x), int(r.center_y)
            cv2.circle(overlay, (cx, cy), 7, WEED_BGR, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # HIGH riskli mahsulden en yakin ota kesikli cizgi
    if (not crops_df.empty and not weeds_df.empty
            and "nearest_weed_id" in crops_df.columns):
        weed_by_id = weeds_df.set_index("det_id")
        for _, r in crops_df[crops_df["risk"] == "HIGH"].iterrows():
            wid = int(r.nearest_weed_id)
            if wid in weed_by_id.index:
                wx = int(weed_by_id.loc[wid, "center_x"])
                wy = int(weed_by_id.loc[wid, "center_y"])
                draw_dashed_line(
                    img,
                    (int(r.center_x), int(r.center_y)),
                    (wx, wy),
                    RISK_COLORS_BGR["HIGH"],
                    thickness=1,
                )

    # Crop kutulari risk renginde
    for _, r in crops_df.iterrows():
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        risk  = r.get("risk", "LOW")
        color = RISK_COLORS_BGR[risk]
        lw    = 3 if risk == "HIGH" else (2 if risk == "MED" else 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

        d = r.get("nearest_weed_px", np.inf)
        dist_txt = f"{int(d)}px" if d != np.inf else "---"
        put_label(img, f"{r['type']} {risk} {dist_txt}", x1, y1, color)

    # Legend (sol ust)
    by_risk = crops_df.groupby("risk").size().to_dict() if not crops_df.empty else {}
    legend_items = [
        (RISK_COLORS_BGR["HIGH"],
         f"YUKSEK (<{RISK_HIGH_PX}px) : {by_risk.get('HIGH', 0)}"),
        (RISK_COLORS_BGR["MED"],
         f"ORTA   ({RISK_HIGH_PX}-{RISK_MED_PX}px): {by_risk.get('MED', 0)}"),
        (RISK_COLORS_BGR["LOW"],
         f"DUSUK  (>{RISK_MED_PX}px)  : {by_risk.get('LOW', 0)}"),
    ]
    for i, (col, txt) in enumerate(legend_items):
        y = 28 + i * 26
        cv2.rectangle(img, (8, y - 12), (22, y + 4), col, -1)
        cv2.putText(img, txt, (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    cv2.imwrite(str(out_path), img)
    print(f"  Risk gorseli    : {out_path.name}")


# ── CSV Ozet ──────────────────────────────────────────────────────────────────

def save_summary(crops_df: pd.DataFrame, weeds_df: pd.DataFrame,
                 img_name: str, out_dir: Path, stem: str) -> dict:

    by_type = crops_df.groupby("type").size().to_dict() if not crops_df.empty else {}
    by_risk = crops_df.groupby("risk").size().to_dict() if not crops_df.empty else {}

    total_crops = len(crops_df)
    total_weeds = len(weeds_df)
    high_risk   = by_risk.get("HIGH", 0)
    risk_pct    = round(high_risk / total_crops * 100, 1) if total_crops else 0.0

    rows = [
        ["=== BIRLESIK TARLA ANALIZI ===", ""],
        ["Goruntu",                        img_name],
        ["", ""],
        ["=== MAHSUL ===",                 ""],
        ["Toplam Mahsul",                  total_crops],
    ] + [[f"  {t}", c] for t, c in sorted(by_type.items())] + [
        ["", ""],
        ["=== OT ===",                     ""],
        ["Toplam Ot",                      total_weeds],
        ["", ""],
        ["=== RISK DAGILIMI ===",          ""],
        [f"YUKSEK RISK (<{RISK_HIGH_PX}px)",           by_risk.get("HIGH", 0)],
        [f"ORTA RISK ({RISK_HIGH_PX}-{RISK_MED_PX}px)", by_risk.get("MED",  0)],
        [f"DUSUK RISK (>{RISK_MED_PX}px)",              by_risk.get("LOW",  0)],
        ["Risk Altindaki Mahsul (%)",      risk_pct],
    ]

    with open(out_dir / f"{stem}_summary.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    # Detay CSV'leri
    if not crops_df.empty:
        keep = [c for c in ["plant_id", "type", "primary", "is_stem", "confidence",
                             "center_x", "center_y", "x1", "y1", "x2", "y2",
                             "nearest_weed_px", "risk"] if c in crops_df.columns]
        crops_df[keep].to_csv(out_dir / f"{stem}_crops.csv", index=False)

    if not weeds_df.empty:
        keep = [c for c in ["det_id", "confidence", "center_x", "center_y",
                             "x1", "y1", "x2", "y2"] if c in weeds_df.columns]
        weeds_df[keep].to_csv(out_dir / f"{stem}_weeds.csv", index=False)

    print(f"  Ozet CSV        : {stem}_summary.csv")

    return {
        "image":      img_name,
        "crop_count": total_crops,
        "weed_count": total_weeds,
        "high_risk":  high_risk,
        "risk_pct":   risk_pct,
    }


def print_summary(s: dict):
    print(f"\n  {'─'*52}")
    print(f"  {s['image']}")
    print(f"  {'─'*52}")
    print(f"  Mahsul           : {s['crop_count']:>5}")
    print(f"  Ot               : {s['weed_count']:>5}")
    print(f"  Yuksek Risk      : {s['high_risk']:>5}  mahsul (<{RISK_HIGH_PX}px)")
    print(f"  Risk Yuzdesi     : %{s['risk_pct']:.1f}")
    print(f"  {'─'*52}")


# ── Ana Akis ──────────────────────────────────────────────────────────────────

def analyze_image(crop_model, weed_model,
                  image_path: Path, base_out: Path) -> dict:
    print(f"\n  Isleniyor: {image_path.name}")
    out_dir = base_out / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    crops_df = run_crop_detection(crop_model, image_path)
    weeds_df = run_weed_detection(weed_model, image_path)
    print(f"    Mahsul: {len(crops_df)}  |  Ot: {len(weeds_df)}")

    crops_df = calculate_risk(crops_df, weeds_df)

    draw_combined(image_path, crops_df, weeds_df,
                  out_dir / f"{stem}_combined.jpg")
    draw_risk(    image_path, crops_df, weeds_df,
                  out_dir / f"{stem}_risk.jpg")

    summary = save_summary(crops_df, weeds_df, image_path.name, out_dir, stem)
    print_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str)
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()

    COMBINED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  BIRLESIK TARLA ANALIZI -- CROP + WEED")
    print("=" * 60)

    crop_path = find_crop_model()
    weed_path = find_weed_model()
    print(f"\n  Crop modeli: {crop_path}")
    print(f"  Weed modeli: {weed_path}")

    crop_model = YOLO(str(crop_path))
    weed_model = YOLO(str(weed_path))

    if args.image:
        images = [Path(args.image)]
    elif args.folder:
        folder = Path(args.folder)
        images = [f for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG")
                  for f in folder.glob(ext)]
        print(f"  {len(images)} goruntu bulundu")
    else:
        print("  Kullanim:")
        print("    python field_combined_analyze.py --image <goruntu.jpg>")
        print("    python field_combined_analyze.py --folder <klasor/>")
        sys.exit(0)

    if not images:
        print("  Goruntu bulunamadi!")
        sys.exit(1)

    summaries = [
        analyze_image(crop_model, weed_model, img, COMBINED_OUT_DIR)
        for img in images
    ]

    if len(images) > 1:
        total_crop = sum(s["crop_count"] for s in summaries)
        total_weed = sum(s["weed_count"] for s in summaries)
        avg_risk   = sum(s["risk_pct"]   for s in summaries) / len(summaries)

        print(f"\n{'='*60}")
        print(f"  TOPLU OZET ({len(images)} goruntu)")
        print(f"  Toplam Mahsul    : {total_crop}")
        print(f"  Toplam Ot        : {total_weed}")
        print(f"  Ort. Risk        : %{avg_risk:.1f}")
        print(f"{'='*60}")

        with open(COMBINED_OUT_DIR / "COMBINED_BATCH.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Goruntu", "Mahsul", "Ot", "YuksekRisk", "Risk%"])
            for s in summaries:
                w.writerow([s["image"], s["crop_count"], s["weed_count"],
                            s["high_risk"], s["risk_pct"]])

    print(f"\n  Ciktilar: {COMBINED_OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
