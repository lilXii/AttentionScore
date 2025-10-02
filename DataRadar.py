#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar charts from windows.csv

- 讀取 <STAMP>__windows.csv（由 123.py 產生）
- 指標 → 場內百分位（方向統一：BCEA/GTE/Blink* 低好；FixRatio 高好）
- 輸出：
  1) <STAMP>__radar_overall.png   （各指標的中位數）
  2) <STAMP>__radar_topK.png      （依 AttentionScore 取前 K 段，每段一個雷達）

依賴：numpy, pandas, matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["BCEA", "GTE", "FixRatio", "BlinkRate", "BlinkFrac"]  # 有缺就自動剔除

def percentile_rank_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    p = np.full_like(v, np.nan, dtype=float)
    finite = np.isfinite(v); idx = np.where(finite)[0]
    if idx.size == 0: return p
    x = v[idx]
    if np.unique(x).size == 1:
        p[idx] = 0.5; return p
    order = np.argsort(x, kind="mergesort")
    ranks_srt = np.arange(1, len(x)+1, dtype=float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        avg = 0.5*(i+1+j)
        ranks_srt[i:j] = avg
        i=j
    ranks = np.empty_like(ranks_srt); ranks[order]=ranks_srt
    p[idx] = ranks/(len(x)+1.0)
    return p

def directional_percentiles(df: pd.DataFrame) -> (list[str], np.ndarray):
    labels = []
    cols = []
    for m in METRICS:
        if m not in df.columns: continue
        p = percentile_rank_vector(df[m].to_numpy(float))
        if m in ("BCEA","GTE","BlinkRate","BlinkFrac"):
            p = 1.0 - p  # 小越好 → 反轉
        # 如果全 NaN，略過該軸
        if not np.isfinite(p).any(): continue
        labels.append(m)
        cols.append(p)
    if not cols:
        raise RuntimeError("沒有可用的指標可畫雷達圖。")
    return labels, np.vstack(cols).T  # shape: (N_windows, N_metrics_kept)

def radar(ax, labels, values, title):
    # values: shape (K,) in [0,1]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close
    v = np.asarray(values, float).tolist()
    v += v[:1]
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.plot(angles, v, linewidth=1.5)
    ax.fill(angles, v, alpha=0.15)
    ax.set_title(title, pad=14)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stamp", help="時間戳（如 20251001_211123），需有對應 <STAMP>__windows.csv")
    ap.add_argument("--topk", type=int, default=5, help="雷達 Top-K 視窗數")
    ap.add_argument("--outdir", default=".", help="輸出目錄")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    win_csv = Path(f"{args.stamp}__windows.csv")
    if not win_csv.exists():
        raise FileNotFoundError(f"找不到 {win_csv.name}，請先跑 123.py 產生 windows.csv")
    df = pd.read_csv(win_csv)

    # 1) 轉成方向一致化的百分位矩陣  (N_windows × N_metrics_kept)
    labels, P = directional_percentiles(df)

    # 2) 整體雷達（各指標中位數）
    overall = np.nanmedian(P, axis=0)
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111, projection='polar')
    radar(ax, labels, overall, f"Overall (median) — {args.stamp}")
    fig.tight_layout()
    out1 = outdir / f"{args.stamp}__radar_overall.png"
    fig.savefig(out1, dpi=150); plt.close(fig)
    print(f"[OK] Radar (overall) → {out1}")

    # 3) Top-K 視窗雷達（依 AttentionScore，若無則依 overall 接近度排序）
    if "AttentionScore" in df.columns and df["AttentionScore"].notna().any():
        top_idx = df["AttentionScore"].to_numpy(float).argsort()[::-1][:args.topk]
    else:
        # 無 AttentionScore 時：用與 overall 的 cosine 相似度挑前 K
        def csim(a,b):
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            return -np.inf if na==0 or nb==0 else float(a@b/(na*nb))
        sims = np.array([csim(P[i], overall) for i in range(P.shape[0])])
        top_idx = np.argsort(sims)[::-1][:args.topk]

    # 畫多個窗的雷達（每窗一個多邊形）
    fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection='polar')
    for rank, i in enumerate(top_idx, 1):
        title = f"Top{rank}: {df.loc[i,'start']:.1f}-{df.loc[i,'end']:.1f}s"
        radar(ax, labels, P[i], title="")
    ax.set_title(f"Top-{len(top_idx)} windows — {args.stamp}", pad=14)
    fig.tight_layout()
    out2 = outdir / f"{args.stamp}__radar_top{len(top_idx)}.png"
    fig.savefig(out2, dpi=150); plt.close(fig)
    print(f"[OK] Radar (top{len(top_idx)}) → {out2}")

if __name__ == "__main__":
    main()
