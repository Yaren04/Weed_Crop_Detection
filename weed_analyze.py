"""
Gorev 4 — Yabanci Ot Tespiti, Konumlandirma ve Yogunluk Analizi

Ciktilar (results/weed/<goruntu_adi>/ altinda):
  _detection.jpg      : Her ot/ekin icin kutu + ID + confidence
  _heatmap.jpg        : Ot yogunluk isi haritasi
  _patches.jpg        : DBSCAN patch/yigin kumesi + sira analizi
  _detections.csv     : ID, sinif, koordinat, alan, confidence, patch_id
  _summary.csv        : Toplam ot, toplam alan, patch sayisi, yogunluk skoru
  _error_analysis.jpg : Ot-ekin karisikligi zor ornekler (bonus)

Kullanim:
    conda activate crop_detection
    python weed_analyze.py                        # test setinden 5 ornek
    python weed_analyze.py --image goruntu.jpg    # tek goruntu
    python weed_analyze.py --folder klasor/       # toplu analiz
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from weed_config import (
    WEED_CLASS_COLORS,
    WEED_CLASS_NAMES,
    WEED_CONF_THRESHOLD,
    WEED_RESULTS_DIR,
    WEED_YOLO_DATASET,
)

# ── Sabitler ─────────────────────────────────────────────────────────────────
DBSCAN_MIN_SAMPLES = 2
HEATMAP_SIGMA      = 30    # Gaussian blur yumusatma
CLUSTER_PALETTE    = [
    "#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6",
    "#1ABC9C","#E67E22","#2980B9","#27AE60","#8E44AD",
    "#F1C40F","#16A085","#D35400","#C0392B","#7F8C8D",
]


# ── Yardimcilar ───────────────────────────────────────────────────────────────

def find_weed_model() -> Path:
    root = Path(__file__).parent
    for pattern in ("weed_yolo/**/weights/best.pt", "weed_yolo/**/weights/last.pt"):
        c = list(root.glob(pattern))
        if c:
            return max(c, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(
        "Weed modeli bulunamadi.\n"
        "Once 'python weed_pipeline.py' ile egitim tamamlayin."
    )


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)


def put_label(img, text, x1, y1, color_bgr):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    y_top = max(y1 - th - 6, 0)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 4, y1), color_bgr, -1)
    cv2.putText(img, text, (x1+2, max(y1-4, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)


def adaptive_eps(df: pd.DataFrame, img_w: int, img_h: int) -> float:
    """Ortalama ot boyutuna gore DBSCAN eps hesapla."""
    weeds = df[df["class"] == "weed"] if "class" in df.columns else df
    if weeds.empty:
        return max(img_w * 0.06, 50)
    diag = np.sqrt(weeds["width_px"]**2 + weeds["height_px"]**2).mean()
    return max(diag * 1.2, img_w * 0.04)


# ── 1. Tespit ─────────────────────────────────────────────────────────────────

def run_detection(model, image_path: Path) -> pd.DataFrame:
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
        cid       = int(box.cls[0])
        cls_name  = WEED_CLASS_NAMES[cid] if cid < len(WEED_CLASS_NAMES) else "unknown"
        conf      = float(box.conf[0])
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        rows.append({
            "det_id":     i + 1,
            "class":      cls_name,
            "confidence": round(conf, 3),
            "center_x":   round((x1+x2)/2, 1),
            "center_y":   round((y1+y2)/2, 1),
            "width_px":   round(x2-x1, 1),
            "height_px":  round(y2-y1, 1),
            "area_px2":   round((x2-x1)*(y2-y1), 1),
            "x1":round(x1,1),"y1":round(y1,1),
            "x2":round(x2,1),"y2":round(y2,1),
            "img_w":      img_w,
            "img_h":      img_h,
        })

    return pd.DataFrame(rows)


# ── 2. Patch Kumesi ───────────────────────────────────────────────────────────

def add_patches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    img_w = int(df["img_w"].iloc[0])
    img_h = int(df["img_h"].iloc[0])

    eps = adaptive_eps(df, img_w, img_h)
    print(f"    DBSCAN eps={eps:.0f}px")

    if len(df) >= DBSCAN_MIN_SAMPLES:
        coords = df[["center_x","center_y"]].values
        labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(coords)
        df["patch_id"] = [
            f"Patch_{l+1}" if l >= 0 else "Izole" for l in labels
        ]
    else:
        df["patch_id"] = "Izole"

    return df


# ── 3. Tespit Gorseli ─────────────────────────────────────────────────────────

def draw_detection(image_path: Path, df: pd.DataFrame, out_path: Path):
    """Her tespit icin ID + sinif + confidence kutusu."""
    img = cv2.imread(str(image_path))
    if img is None: return

    weed_color = hex_to_bgr(WEED_CLASS_COLORS["weed"])

    for _, r in df.iterrows():
        x1,y1,x2,y2 = int(r.x1),int(r.y1),int(r.x2),int(r.y2)

        cv2.rectangle(img, (x1,y1), (x2,y2), weed_color, 2)
        cx,cy = int(r.center_x), int(r.center_y)
        cv2.circle(img, (cx,cy), 3, weed_color, -1)

        label = f"#{int(r.det_id)} {r.confidence:.2f}"
        put_label(img, label, x1, y1, weed_color)

    # Sol ust ozet
    cv2.putText(img, f"Ot: {len(df)}", (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, weed_color, 2)

    cv2.imwrite(str(out_path), img)
    print(f"  Tespit gorseli  : {out_path.name}")


# ── 4. Yogunluk Isi Haritasi ─────────────────────────────────────────────────

def draw_heatmap(image_path: Path, df: pd.DataFrame, out_path: Path):
    """Ot yogunlugunu Gaussian kernel ile isi haritasi olarak goster."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None: return

    img_h, img_w = img_bgr.shape[:2]
    heatmap      = np.zeros((img_h, img_w), dtype=np.float32)

    weeds = df[df["class"] == "weed"]
    for _, r in weeds.iterrows():
        cx, cy = int(r.center_x), int(r.center_y)
        if 0 <= cx < img_w and 0 <= cy < img_h:
            # Buyuk ot = daha guclu sinyal (alan ile agirlandir)
            weight = max(r.area_px2 / (img_w * img_h) * 1000, 1.0)
            heatmap[cy, cx] += weight

    # Gaussian blur ile yumusat
    sigma    = max(HEATMAP_SIGMA, int(img_w * 0.03))
    heatmap  = gaussian_filter(heatmap, sigma=sigma)

    # Normalize → renk haritasi
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), alpha=0.6)
    hm = ax.imshow(heatmap, cmap="RdYlGn_r", alpha=0.55,
                   vmin=0, vmax=1, interpolation="bilinear")
    plt.colorbar(hm, ax=ax, label="Ot Yogunlugu (normalize)")

    # Ot merkezlerini nokta olarak goster
    if not weeds.empty:
        ax.scatter(weeds["center_x"], weeds["center_y"],
                   c="white", s=15, linewidths=0.5, edgecolors="black", zorder=5)

    # Yogunluk skoru hesapla
    coverage_pct = float((heatmap > 0.1).sum()) / (img_w * img_h) * 100
    ax.set_title(
        f"Ot Yogunluk Haritasi  |  {len(weeds)} ot tespit  |  "
        f"Alan kaplanimi: %{coverage_pct:.1f}",
        fontsize=11, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Isi haritasi    : {out_path.name}")


# ── 5. Patch Gorseli ─────────────────────────────────────────────────────────

def draw_patches(image_path: Path, df: pd.DataFrame, out_path: Path):
    """DBSCAN patch kumeleri farkli renklerle."""
    img_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    if img_rgb is None: return

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(img_rgb)
    ax.set_title(f"Ot Patch Analizi — {image_path.name}",
                 fontsize=11, fontweight="bold")

    patches_col = "patch_id" if "patch_id" in df.columns else None
    if patches_col:
        patch_ids = df[patches_col].unique()
        cmap = {}
        pi   = 0
        for pid in sorted(patch_ids):
            cmap[pid] = "#AAAAAA" if pid == "Izole" else CLUSTER_PALETTE[pi % len(CLUSTER_PALETTE)]
            if pid != "Izole": pi += 1

    legend_items = []
    for _, r in df.iterrows():
        x1,y1,x2,y2 = r.x1,r.y1,r.x2,r.y2

        pid   = r.get(patches_col, "Izole") if patches_col else "Izole"
        color = cmap.get(pid, "#AAAAAA")

        rect = plt.Rectangle((x1,y1), x2-x1, y2-y1,
                              lw=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text((x1+x2)/2, y1-5, f"#{int(r.det_id)}",
                color=color, fontsize=7, ha="center", fontweight="bold")

    # Legend
    if patches_col:
        for pid, color in cmap.items():
            cnt = len(df[df[patches_col]==pid])
            legend_items.append(
                mpatches.Patch(color=color, label=f"{pid} ({cnt} ot)")
            )
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", fontsize=8,
                  framealpha=0.85, title="Patch Kumeleri")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Patch gorseli   : {out_path.name}")


# ── 6. Hata Analizi (Bonus) ──────────────────────────────────────────────────

def draw_error_analysis(image_path: Path, df: pd.DataFrame, out_path: Path):
    """
    Ot-Ekin Karisikligi Analizi:
    Dusuk confidence'li tespitler + ot ile ekin cakisan kutular
    """
    if df.empty: return

    img = cv2.imread(str(image_path))
    if img is None: return

    # Dusuk confidence (0.4-0.6 arasi = model belirsiz)
    low_conf    = df[df["confidence"] < 0.6]
    # Birbirine cok yakin / ust uste tespitler (potansiyel duplicate)
    overlap_ids = set()
    for i, r1 in df.iterrows():
        for j, r2 in df.iterrows():
            if i >= j: continue
            ix1 = max(r1.x1, r2.x1); iy1 = max(r1.y1, r2.y1)
            ix2 = min(r1.x2, r2.x2); iy2 = min(r1.y2, r2.y2)
            if ix2 > ix1 and iy2 > iy1:
                iou = (ix2-ix1)*(iy2-iy1) / min(
                    (r1.x2-r1.x1)*(r1.y2-r1.y1),
                    (r2.x2-r2.x1)*(r2.y2-r2.y1)
                )
                if iou > 0.4:   # %40'tan fazla cakisma = muhtemelen duplicate
                    overlap_ids.add(int(r1.det_id))
                    overlap_ids.add(int(r2.det_id))

    has_issue = len(low_conf) > 0 or len(overlap_ids) > 0

    for _, r in df.iterrows():
        x1,y1,x2,y2 = int(r.x1),int(r.y1),int(r.x2),int(r.y2)
        det_id       = int(r.det_id)

        if det_id in overlap_ids:
            color = (255, 165, 0)   # Turuncu — cakisan kutu
            lw    = 3
            tag   = "CAKISMA"
        elif r.confidence < 0.6:
            color = (128, 0, 128)   # Mor — dusuk confidence
            lw    = 2
            tag   = f"DUSUK {r.confidence:.2f}"
        else:
            color = hex_to_bgr(WEED_CLASS_COLORS["weed"])
            lw    = 1
            tag   = None

        cv2.rectangle(img, (x1,y1), (x2,y2), color, lw)
        if tag:
            put_label(img, tag, x1, y1, color)

    if not has_issue:
        cv2.putText(img, "Hata yok", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 2)

    cv2.imwrite(str(out_path), img)
    print(f"  Hata analizi    : {out_path.name}")
    return len(low_conf), len(overlap_ids)


# ── 7. CSV + Ozet ─────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, img_name: str,
                 out_dir: Path, stem: str) -> dict:
    img_w = int(df["img_w"].iloc[0]) if not df.empty else 1
    img_h = int(df["img_h"].iloc[0]) if not df.empty else 1

    by_patch    = df.groupby("patch_id").size().to_dict() \
                  if "patch_id" in df.columns else {}
    total_area  = df["area_px2"].sum()
    img_area    = img_w * img_h
    density_pct = total_area / img_area * 100

    df.to_csv(out_dir / f"{stem}_detections.csv", index=False)

    rows = [
        ["=== OT TESPITI OZETI ===",    ""],
        ["Goruntu",                      img_name],
        ["Toplam Ot (weed)",             len(df)],
        ["Ort. Confidence",              round(df["confidence"].mean(),3)
                                         if not df.empty else 0],
        ["Toplam Ot Alani (px2)",        round(total_area, 1)],
        ["Yogunluk Skoru (%)",           round(density_pct, 3)],
        ["Goruntu Alani (px2)",          img_area],
        ["", ""],
        ["=== PATCH DAGILIMI ===",       ""],
        ["Patch",                        "Ot Sayisi"],
    ] + sorted(by_patch.items())

    with open(out_dir / f"{stem}_summary.csv","w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"  Detay CSV       : {stem}_detections.csv")
    print(f"  Ozet CSV        : {stem}_summary.csv")

    return {
        "image":       img_name,
        "weed_count":  len(df),
        "total_area":  round(total_area, 1),
        "density_pct": round(density_pct, 3),
        "patch_count": len([p for p in by_patch if p != "Izole"]),
        "avg_conf":    round(df["confidence"].mean(),3) if not df.empty else 0,
    }


def print_summary(s: dict):
    print(f"\n  {'─'*52}")
    print(f"  {s['image']}")
    print(f"  {'─'*52}")
    print(f"  Toplam Ot         : {s['weed_count']:>5}  <- birincil metrik")
    print(f"  Toplam Ot Alani   : {s['total_area']:>10.0f} px2")
    print(f"  Yogunluk Skoru    : %{s['density_pct']:.3f}")
    print(f"  Patch Sayisi      : {s['patch_count']:>5}")
    print(f"  Ort. Confidence   : {s['avg_conf']}")
    print(f"  {'─'*52}")


# ── Ana Akis ──────────────────────────────────────────────────────────────────

def analyze_image(model, image_path: Path, base_out: Path) -> dict:
    print(f"\n  Isleniyor: {image_path.name}")
    out_dir = base_out / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    df = run_detection(model, image_path)
    if df.empty:
        print("    Hic tespit yok.")
        return {"image": image_path.name, "weed_count":0,
                "total_area":0, "density_pct":0, "patch_count":0, "avg_conf":0}

    df = add_patches(df)

    draw_detection(    image_path, df, out_dir / f"{stem}_detection.jpg")
    draw_heatmap(      image_path, df, out_dir / f"{stem}_heatmap.jpg")
    draw_patches(      image_path, df, out_dir / f"{stem}_patches.jpg")
    draw_error_analysis(image_path, df, out_dir / f"{stem}_error_analysis.jpg")

    summary = save_outputs(df, image_path.name, out_dir, stem)
    print_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str)
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()

    out_dir = WEED_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  GOREV 4 — YABANCI OT TESPITI & YOGUNLUK ANALIZI")
    print("="*60)

    model_path = find_weed_model()
    print(f"\n  Model: {model_path}")
    model = YOLO(str(model_path))

    if args.image:
        images = [Path(args.image)]
    elif args.folder:
        folder = Path(args.folder)
        images = [f for ext in ("*.jpg","*.jpeg","*.png","*.JPG")
                  for f in folder.glob(ext)]
        print(f"  {len(images)} goruntu bulundu")
    else:
        test_dir = Path(__file__).parent / "weed_yolo_dataset" / "images" / "test"
        images   = list(test_dir.glob("*.jpg"))[:5]
        print(f"  Test setinden {len(images)} goruntu (--image veya --folder ile degistir)")

    if not images:
        print("  Goruntu bulunamadi!")
        sys.exit(1)

    summaries = [analyze_image(model, img, out_dir) for img in images]

    # Toplu ozet
    if len(images) > 1:
        total_weed  = sum(s["weed_count"]  for s in summaries)
        total_area  = sum(s["total_area"]  for s in summaries)
        avg_density = sum(s["density_pct"] for s in summaries) / len(summaries)

        print(f"\n{'='*60}")
        print(f"  TOPLU OZET ({len(images)} goruntu)")
        print(f"  Toplam Ot         : {total_weed}")
        print(f"  Toplam Ot Alani   : {total_area:.0f} px2")
        print(f"  Ort. Yogunluk     : %{avg_density:.3f}")
        print(f"{'='*60}")

        with open(out_dir / "WEED_BATCH_SUMMARY.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Goruntu","Ot","Alan_px2","Yogunluk_%","Patch","Conf"])
            for s in summaries:
                w.writerow([s["image"], s["weed_count"],
                            s["total_area"], s["density_pct"],
                            s["patch_count"], s["avg_conf"]])

    print(f"\n  Ciktilar: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
