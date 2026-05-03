"""
Gorev 3 — Yeni Ekilmis Urun Tespiti, Sayimi ve Grup Analizi

Oncelik:
  1. Birincil tespit  → "Fide / Yeni Ekin"  (tur bilinmese de tespit edilir)
  2. Ikincil bilgi    → Tur: Misir / Fasulye / Pirasa  (model taniyorsa gosterilir)
  3. Ek bilgi         → Govde tespiti (ayri renkle, birincil sayima dahil degil)

Kullanim:
    conda activate crop_detection

    python analyze_field.py                         # test setinden 5 ornek
    python analyze_field.py --image goruntu.jpg     # tek goruntu
    python analyze_field.py --folder klasor/        # klasordeki tum gorseller

Ciktilar  →  results/field_analysis/<goruntu_adi>/
    _detection.jpg   : Fide kutulari (kirmizi=fide, gri=govde) + ID
    _type.jpg        : Tur renkleriyle (misir/fasulye/pirasa)
    _clusters.jpg    : DBSCAN kume + sira analizi
    _detections.csv  : Her tespit icin satir (ID, birincil, tur, koordinat, boyut, kume)
    _summary.csv     : Toplam / tur / kume / sira ozeti
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
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from config import (
    CLASS_HIERARCHY,
    CLASS_NAMES,
    CONF_THRESHOLD,
    PRIMARY_COLORS,
    RESULTS_DIR,
    TYPE_COLORS,
)

# ── Sabitler ──────────────────────────────────────────────────────────────────
# NOT: DBSCAN_EPS artik sabit degil — goruntudeki ortalama bitki boyutuna gore
#      otomatik hesaplaniyor (bkz. adaptive_eps fonksiyonu).
#      Manuel override icin asagidaki degeri None yerine bir sayi yaz.
DBSCAN_EPS_OVERRIDE = None  # ornek: 200  — None ise otomatik
DBSCAN_MIN_SAMPLES  = 2     # kume icin min bitki sayisi
# Sira toleransi da otomatik: ortalama bitki yuksekliginin %60'i
ROW_TOLERANCE_OVERRIDE = None  # None ise otomatik

CLUSTER_PALETTE = [
    "#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6",
    "#1ABC9C","#E67E22","#2980B9","#27AE60","#8E44AD",
    "#F1C40F","#16A085","#D35400","#C0392B","#7F8C8D",
]

# ── Yardimcilar ───────────────────────────────────────────────────────────────

def adaptive_eps(df: pd.DataFrame, img_w: int, img_h: int) -> tuple[float, float]:
    """
    Goruntudeki ortalama bitki boyutuna gore DBSCAN eps ve sira toleransini hesapla.

    Mantik:
      - eps      = ortalama kosegen * 1.0
                   Yani: iki bitki merkezi arasindaki mesafe bir bitki boyutundan
                   fazlaysa ayri kume sayilir. Gercek tarim gruplarini iyi ayirir.
      - row_tol  = ortalama yukseklik * 0.8
                   Ayni sirada bitkiler bu kadar yukari/asagi sapabilir.

    Not: DBSCAN_EPS_OVERRIDE veya ROW_TOLERANCE_OVERRIDE ile manuel override edilebilir.
    """
    if df.empty:
        return max(img_w * 0.05, 60), max(img_h * 0.04, 40)

    fide_df = df[~df["is_stem"]] if "is_stem" in df.columns else df
    if fide_df.empty:
        fide_df = df

    diag  = np.sqrt(fide_df["width_px"]**2 + fide_df["height_px"]**2).mean()
    avg_h = fide_df["height_px"].mean()

    # 1.0: iki bitki arasindaki merkez mesafesi bir bitki boyutunu gecerse = ayri kume
    eps     = diag * 1.0
    row_tol = avg_h * 0.8

    if DBSCAN_EPS_OVERRIDE is not None:
        eps = DBSCAN_EPS_OVERRIDE
    if ROW_TOLERANCE_OVERRIDE is not None:
        row_tol = ROW_TOLERANCE_OVERRIDE

    return round(eps, 1), round(row_tol, 1)


def find_best_model() -> Path:
    root = Path(__file__).parent
    for pattern in ("weights/best.pt", "weights/last.pt"):
        c = list(root.rglob(pattern))
        if c:
            return max(c, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(
        "Model bulunamadi. Once 'python main.py' ile egitim tamamlayin."
    )


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)


def put_label(img, text, x1, y1, color_bgr):
    """Kutunun ustune renkli etiket yaz."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color_bgr, -1)
    cv2.putText(img, text, (x1 + 2, max(y1 - 4, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)


# ── 1. Tespit ─────────────────────────────────────────────────────────────────

def run_detection(model, image_path: Path) -> pd.DataFrame:
    result = model.predict(source=str(image_path), conf=CONF_THRESHOLD, verbose=False)[0]

    rows = []
    if result.boxes is None:
        return pd.DataFrame()

    img_h, img_w = result.orig_shape
    for i, box in enumerate(result.boxes):
        raw_cls  = CLASS_NAMES[int(box.cls[0])]
        primary, tur, is_stem = CLASS_HIERARCHY.get(raw_cls, ("Fide", "Bilinmiyor", False))
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        rows.append({
            "plant_id":      i + 1,
            "primary":       primary,             # "Fide" veya "Govde"
            "type":          tur,                 # "Misir", "Fasulye", "Pirasa"
            "raw_class":     raw_cls,
            "is_stem":       is_stem,
            "confidence":    round(float(box.conf[0]), 3),
            "center_x":      round((x1+x2)/2, 1),
            "center_y":      round((y1+y2)/2, 1),
            "width_px":      round(x2-x1, 1),
            "height_px":     round(y2-y1, 1),
            "area_px2":      round((x2-x1)*(y2-y1), 1),
            "x1":round(x1,1),"y1":round(y1,1),
            "x2":round(x2,1),"y2":round(y2,1),
            "img_w":         img_w,
            "img_h":         img_h,
        })

    return pd.DataFrame(rows)


# ── 2. Kumeleme ───────────────────────────────────────────────────────────────

def add_clusters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    img_w = int(df["img_w"].iloc[0]) if "img_w" in df.columns else 1280
    img_h = int(df["img_h"].iloc[0]) if "img_h" in df.columns else 960

    eps, row_tol = adaptive_eps(df, img_w, img_h)
    print(f"    DBSCAN eps={eps:.0f}px  |  sira toleransi={row_tol:.0f}px  "
          f"(goruntu {img_w}x{img_h})")

    # Sadece Fide'leri kumele
    fide_idx = df[~df["is_stem"]].index
    if len(fide_idx) >= DBSCAN_MIN_SAMPLES:
        coords = df.loc[fide_idx, ["center_x","center_y"]].values
        labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(coords)
        df.loc[fide_idx, "cluster"] = [
            f"Kume_{l+1}" if l >= 0 else "Izole" for l in labels
        ]
    else:
        df.loc[fide_idx, "cluster"] = "Izole"
    df["cluster"] = df["cluster"].fillna("(Govde)")

    # Sira analizi — sadece en az 2 fide iceren satirlari goster
    # (tek tespitten olusan "sahte sira" gorunumunu onler)
    sorted_cy = sorted(df["center_y"].unique())
    row_labels, current_row, prev = {}, 1, None
    for cy in sorted_cy:
        if prev is None or cy - prev > row_tol:
            current_row += (0 if prev is None else 1)
        row_labels[cy] = f"Sira_{current_row}"
        prev = cy

    df["row_id"] = df["center_y"].map(
        lambda cy: row_labels[min(row_labels, key=lambda k: abs(k-cy))]
    )

    # Tek elemanli satirlari "Izole_Sira" olarak isaretle
    row_counts = df.groupby("row_id")["row_id"].transform("count")
    df.loc[row_counts < 2, "row_id"] = df.loc[row_counts < 2, "row_id"].str.replace(
        "Sira_", "IzoleSira_"
    )

    return df


# ── 3. Gorseller ──────────────────────────────────────────────────────────────

def draw_primary(image_path: Path, df: pd.DataFrame, out_path: Path):
    """
    Birincil gorsel:
      - Kirmizi kutu + kalin cerceve  = Fide / Yeni Ekin
      - Gri ince cerceve              = Govde (ek bilgi)
      Etiket:  #ID  Fide (Misir)  0.93
    """
    img = cv2.imread(str(image_path))
    if img is None: return

    fide_count = 0
    for _, r in df.iterrows():
        x1,y1,x2,y2 = int(r.x1),int(r.y1),int(r.x2),int(r.y2)

        if r.is_stem:
            # Govde → gri ince cizgi, etiket yok (gorsel karismasi onler)
            cv2.rectangle(img,(x1,y1),(x2,y2),(150,150,150),1)
            cv2.putText(img, f"govde({r.type})",
                        (x1+2,y2-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,(180,180,180),1)
        else:
            # Fide → kalin kirmizi kutu
            color = hex_to_bgr(PRIMARY_COLORS["Fide"])
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            # Merkez noktasi
            cx,cy = int(r.center_x), int(r.center_y)
            cv2.circle(img,(cx,cy),4,color,-1)
            # Etiket: #ID  Fide (Tur)  conf
            label = f"#{int(r.plant_id)} Fide ({r.type})  {r.confidence:.2f}"
            put_label(img, label, x1, y1, color)
            fide_count += 1

    # Kose ozeti
    summary_txt = f"Fide: {fide_count}"
    cv2.putText(img, summary_txt, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                hex_to_bgr(PRIMARY_COLORS["Fide"]), 2)

    cv2.imwrite(str(out_path), img)
    print(f"  Birincil gorsel : {out_path.name}")


def draw_type(image_path: Path, df: pd.DataFrame, out_path: Path):
    """
    Tur gorseli: her tur farkli renk.
      Misir=sari, Fasulye=yesil, Pirasa=mavi
      Govdeler ayni renkle ama noktalı cerceve ile.
    """
    img = cv2.imread(str(image_path))
    if img is None: return

    type_counts = {}
    for _, r in df.iterrows():
        x1,y1,x2,y2 = int(r.x1),int(r.y1),int(r.x2),int(r.y2)
        color = hex_to_bgr(TYPE_COLORS.get(r.type, "#AAAAAA"))

        if r.is_stem:
            cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        else:
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            label = f"#{int(r.plant_id)} {r.type}"
            put_label(img, label, x1, y1, color)
            type_counts[r.type] = type_counts.get(r.type, 0) + 1

    # Kose ozeti
    y_off = 30
    for tur, cnt in type_counts.items():
        color = hex_to_bgr(TYPE_COLORS.get(tur, "#AAAAAA"))
        cv2.putText(img, f"{tur}: {cnt}", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_off += 30

    cv2.imwrite(str(out_path), img)
    print(f"  Tur gorseli     : {out_path.name}")


def draw_clusters(image_path: Path, df: pd.DataFrame, out_path: Path):
    """Sol: DBSCAN kumeleri  |  Sag: sira analizi"""
    img_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    if img_rgb is None: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Küme & Sıra Analizi — {image_path.name}",
                 fontsize=12, fontweight="bold")

    # ── Sol: DBSCAN ──
    ax1.imshow(img_rgb)
    ax1.set_title("DBSCAN Kümeleri (Fide bazlı)", fontweight="bold")

    fide_df  = df[~df["is_stem"]]
    stem_df  = df[df["is_stem"]]
    clusters = fide_df["cluster"].unique() if "cluster" in fide_df.columns else []
    cmap     = {}
    pi       = 0
    for c in sorted(clusters):
        cmap[c] = "#AAAAAA" if c in ("Izole","(Govde)") else CLUSTER_PALETTE[pi % len(CLUSTER_PALETTE)]
        if c not in ("Izole","(Govde)"): pi += 1

    # Govdeleri soluk gri ciz
    for _, r in stem_df.iterrows():
        rect = plt.Rectangle((r.x1,r.y1),r.x2-r.x1,r.y2-r.y1,
                              lw=1, edgecolor="#CCCCCC", facecolor="none", linestyle=":")
        ax1.add_patch(rect)

    # Fideleri kume rengiyle ciz
    for _, r in fide_df.iterrows():
        c     = r.get("cluster","Izole")
        color = cmap.get(c,"#AAAAAA")
        rect  = plt.Rectangle((r.x1,r.y1),r.x2-r.x1,r.y2-r.y1,
                               lw=2, edgecolor=color, facecolor="none")
        ax1.add_patch(rect)
        ax1.text((r.x1+r.x2)/2, r.y1-5, f"#{int(r.plant_id)}",
                 color=color, fontsize=7, ha="center", fontweight="bold")

    legend = [mpatches.Patch(color=c,
              label=f"{k} ({len(fide_df[fide_df['cluster']==k])})")
              for k, c in cmap.items() if k != "(Govde)"]
    ax1.legend(handles=legend, loc="upper right", fontsize=8,
               framealpha=0.85, title="Kümeler")
    ax1.axis("off")

    # ── Sag: Sira ──
    ax2.imshow(img_rgb)
    ax2.set_title("Sıra Analizi (Y koordinatı bazlı)", fontweight="bold")

    if "row_id" in df.columns:
        row_ids   = sorted(df["row_id"].unique(),
                           key=lambda r: int(r.split("_")[1]) if "_" in r else 0)
        row_cmap  = {r: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                     for i, r in enumerate(row_ids)}

        for _, r in df.iterrows():
            rid   = r.get("row_id","?")
            color = row_cmap.get(rid,"#AAAAAA")
            lw    = 2 if not r.is_stem else 1
            rect  = plt.Rectangle((r.x1,r.y1),r.x2-r.x1,r.y2-r.y1,
                                   lw=lw, edgecolor=color, facecolor="none")
            ax2.add_patch(rect)

        for rid in row_ids:
            avg_y = df[df["row_id"]==rid]["center_y"].mean()
            color = row_cmap[rid]
            ax2.axhline(avg_y, color=color, lw=1, ls="--", alpha=0.55)
            ax2.text(8, avg_y-10, f"{rid} ({len(df[df['row_id']==rid])})",
                     color=color, fontsize=8, fontweight="bold",
                     bbox=dict(facecolor="white", alpha=0.6, pad=2))

        row_legend = [mpatches.Patch(color=c, label=f"{r}")
                      for r,c in row_cmap.items()]
        ax2.legend(handles=row_legend, loc="upper right", fontsize=8,
                   framealpha=0.85, title="Sıralar")

    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Kume gorseli    : {out_path.name}")


# ── 4. CSV + Ozet ─────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, image_name: str, out_dir: Path, stem: str) -> dict:
    fide_df = df[~df["is_stem"]]
    stem_df = df[df["is_stem"]]
    total   = len(df)

    by_type    = fide_df.groupby("type").size().to_dict()
    by_cluster = fide_df.groupby("cluster").size().to_dict() if "cluster" in fide_df.columns else {}
    by_row     = df.groupby("row_id").size().to_dict()       if "row_id"  in df.columns    else {}

    # Detay CSV
    df.to_csv(out_dir / f"{stem}_detections.csv", index=False)

    # Ozet CSV
    rows = [
        ["=== BIRINCIL TESPIT ===", ""],
        ["Goruntu",                 image_name],
        ["Toplam Fide (Yeni Ekin)", len(fide_df)],
        ["Toplam Govde Tespiti",    len(stem_df)],
        ["Toplam Tespit",           total],
        ["Ort. Confidence",         round(df["confidence"].mean(),3) if not df.empty else 0],
        ["Ort. Fide Alani (px2)",   round(fide_df["area_px2"].mean(),1) if not fide_df.empty else 0],
        ["",""],
        ["=== TUR DAGILIMI (Fide) ===",""],
        ["Tur","Sayi"],
    ] + [[t, by_type.get(t,0)] for t in ["Misir","Fasulye","Pirasa","Bilinmiyor"]]

    if by_cluster:
        rows += [["",""], ["=== KUME ANALIZI (DBSCAN) ===",""], ["Kume","Fide Sayisi"]]
        rows += sorted(by_cluster.items())

    if by_row:
        rows += [["",""], ["=== SIRA ANALIZI ===",""], ["Sira","Tespit Sayisi"]]
        rows += sorted(by_row.items(), key=lambda x: int(x[0].split("_")[1]) if "_" in x[0] else 0)

    with open(out_dir / f"{stem}_summary.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"  Detay CSV       : {stem}_detections.csv")
    print(f"  Ozet CSV        : {stem}_summary.csv")

    return {
        "image":       image_name,
        "fide":        len(fide_df),
        "stem":        len(stem_df),
        "total":       total,
        "by_type":     by_type,
        "by_cluster":  by_cluster,
        "by_row":      by_row,
        "avg_conf":    round(df["confidence"].mean(),3) if not df.empty else 0,
    }


def print_summary(s: dict):
    print(f"\n  {'─'*52}")
    print(f"  {s['image']}")
    print(f"  {'─'*52}")
    print(f"  Fide (Yeni Ekin) : {s['fide']:>4}  ← birincil metrik")
    print(f"  Govde tespiti    : {s['stem']:>4}  (ek bilgi)")
    print(f"  Ort. Confidence  : {s['avg_conf']}")
    print(f"\n  TUR DAGILIMI:")
    for tur in ["Misir","Fasulye","Pirasa","Bilinmiyor"]:
        cnt = s["by_type"].get(tur,0)
        if cnt:
            bar = "█" * min(cnt,30)
            print(f"    {tur:<12}: {cnt:>4}  {bar}")
    if s["by_cluster"]:
        print(f"\n  KUME (DBSCAN):")
        for k,v in sorted(s["by_cluster"].items()):
            print(f"    {k:<16}: {v}")
    if s["by_row"]:
        print(f"\n  SIRALAR:")
        for k,v in sorted(s["by_row"].items(),
                          key=lambda x: int(x[0].split("_")[1]) if "_" in x[0] else 0):
            print(f"    {k:<12}: {v}")
    print(f"  {'─'*52}")


# ── Ana Akis ──────────────────────────────────────────────────────────────────

def analyze_image(model, image_path: Path, base_out: Path) -> dict:
    print(f"\n  Isleniyor: {image_path.name}")
    out_dir = base_out / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_detection(model, image_path)

    if df.empty:
        print("    Hic tespit yok.")
        return {"image": image_path.name, "fide":0, "stem":0, "total":0,
                "by_type":{},"by_cluster":{},"by_row":{},"avg_conf":0}

    df = add_clusters(df)

    stem = image_path.stem
    draw_primary( image_path, df, out_dir / f"{stem}_detection.jpg")
    draw_type(    image_path, df, out_dir / f"{stem}_type.jpg")
    draw_clusters(image_path, df, out_dir / f"{stem}_clusters.jpg")

    summary = save_outputs(df, image_path.name, out_dir, stem)
    print_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str)
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "field_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  GOREV 3 — YENI EKILMIS URUN TESPITI & GRUP ANALIZI")
    print("  Oncelik: Fide/Yeni Ekin  |  Tur bilgisi: Misir/Fasulye/Pirasa")
    print("="*60)

    model_path = find_best_model()
    print(f"\n  Model: {model_path}")
    model = YOLO(str(model_path))

    if args.image:
        images = [Path(args.image)]
    elif args.folder:
        folder = Path(args.folder)
        images = [f for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG")
                  for f in folder.glob(ext)]
        print(f"  {len(images)} goruntu bulundu")
    else:
        test_dir = Path(__file__).parent / "yolo_dataset" / "images" / "test"
        images   = list(test_dir.glob("*.jpg"))[:5]
        print(f"  Test setinden {len(images)} goruntu (--image veya --folder ile degistir)")

    if not images:
        print("  Goruntu bulunamadi!")
        sys.exit(1)

    summaries = [analyze_image(model, img, out_dir) for img in images]

    # Cok goruntu → toplu ozet
    if len(images) > 1:
        total_fide = sum(s["fide"]  for s in summaries)
        total_stem = sum(s["stem"]  for s in summaries)
        print(f"\n{'='*60}")
        print(f"  TOPLU OZET  ({len(images)} goruntu)")
        print(f"  Toplam Fide  : {total_fide}")
        print(f"  Toplam Govde : {total_stem}")
        print(f"{'='*60}")

        with open(out_dir / "BATCH_SUMMARY.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Goruntu","Fide","Govde","Toplam","Ort_Conf",
                        "Misir","Fasulye","Pirasa"])
            for s in summaries:
                w.writerow([
                    s["image"], s["fide"], s["stem"], s["total"], s["avg_conf"],
                    s["by_type"].get("Misir",0),
                    s["by_type"].get("Fasulye",0),
                    s["by_type"].get("Pirasa",0),
                ])
        print(f"  BATCH_SUMMARY.csv kaydedildi: {out_dir}")

    print(f"\n  Ciktilar: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
