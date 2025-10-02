#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AOI Occupancy Visualization (Center-5 + GridFill)

- 讀取 video_eye_*.csv（指定 STAMP 或自動找最新）
- 將 3D 注視點投影到固定 Unity 螢幕平面 → UV ∈ [0,1]^2
- 僅統計 on-screen 樣本
- 輸出：
  1) <STAMP>__aoi_center5.png          （中心五區 16:9 畫面填色，內嵌百分比）
  2) <STAMP>__aoi_gridfill_<Nx>x<Ny>.png（NX×NY 16:9 畫面填色、畫格線、內嵌百分比）
  3) <STAMP>__aoi_center5.csv          （中心五區百分比）

依賴：numpy, pandas, matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# ---- 固定 Unity 螢幕平面 ----
DEFAULT_PLANE = {
    "center": {"x": 0.33000001311302187, "y": 1.2999999523162842, "z": 4.929999828338623},
    "right":  {"x": 1.0, "y": 0.0, "z": 0.0},
    "up":     {"x": 0.0, "y": 1.0, "z": 0.0},
    "width":  7.950000286102295,
    "height": 4.471874713897705,
}
EPS = 0.05

def _v(d): return np.array([d["x"], d["y"], d["z"]], float)
def get_screen_basis(pdct=DEFAULT_PLANE):
    C = _v(pdct["center"])
    R = _v(pdct["right"]); R /= (np.linalg.norm(R)+1e-12)
    U = _v(pdct["up"]);    U /= (np.linalg.norm(U)+1e-12)
    N = np.cross(R, U);    N /= (np.linalg.norm(N)+1e-12)
    W = float(pdct["width"]); H = float(pdct["height"])
    return C,R,U,N,W,H

def detect_latest_eye(root: Path, stamp: str|None) -> Path:
    if stamp:
        p = root / f"video_eye_{stamp}.csv"
        if not p.exists(): raise FileNotFoundError(f"找不到 {p.name}")
        return p
    eyes = sorted(root.glob("video_eye_*.csv"))
    if not eyes: raise FileNotFoundError("找不到任何 video_eye_*.csv")
    return eyes[-1]

def rays_to_plane_intersection(O, D, C, N):
    denom = (D @ N)
    mask = np.abs(denom) > 1e-12
    t = np.full(O.shape[0], np.nan, float)
    if mask.any():
        t[mask] = ((C - O[mask]) @ N) / denom[mask]
        t[t < 0] = np.nan
    return O + D * t[:,None]

def project_uv(P, C, R, U, N, W, H):
    if (W<=1e-9) or (H<=1e-9): raise ValueError("Invalid screen size")
    p = P - C[None,:]
    u = (p @ R)/W + 0.5
    v = (p @ U)/H + 0.5
    dplane = p @ N
    return u, v, dplane

def quantize_uv(u, v, nx, ny):
    xi = np.clip(np.floor(u*nx).astype(int), 0, nx-1)
    yi = np.clip(np.floor(v*ny).astype(int), 0, ny-1)
    return yi, xi  # row, col

# ------- 中心五區標記 -------
def label_center5(u_on: np.ndarray, v_on: np.ndarray, center_frac: float=0.6):
    cf = float(center_frac)
    cf = min(max(cf, 0.05), 0.95)
    u_min, u_max = 0.5 - cf/2.0, 0.5 + cf/2.0
    v_min, v_max = 0.5 - cf/2.0, 0.5 + cf/2.0

    lab = np.full(u_on.shape, -1, int)
    center_mask = (u_on>=u_min) & (u_on<=u_max) & (v_on>=v_min) & (v_on<=v_max)
    lab[center_mask] = 0
    rest = ~center_mask
    top_mask    = rest & (v_on >  v_max)
    bottom_mask = rest & (v_on <  v_min)
    lab[top_mask] = 1
    lab[bottom_mask] = 2
    rest2 = rest & ~(top_mask | bottom_mask)
    left_mask  = rest2 & (u_on <  u_min)
    right_mask = rest2 & (u_on >  u_max)
    lab[left_mask]  = 3
    lab[right_mask] = 4
    lab[lab<0] = 0
    return lab, (u_min, u_max, v_min, v_max)

def luminance(rgb):
    r,g,b = rgb[:3]
    return 0.2126*r + 0.7152*g + 0.0722*b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stamp", nargs="?", default=None, help="時間戳（如 20251001_211123），省略自動找最新")
    ap.add_argument("--nx", type=int, default=4, help="AOI 水平分割數")
    ap.add_argument("--ny", type=int, default=4, help="AOI 垂直分割數")
    ap.add_argument("--center-frac", type=float, default=0.6, help="中心區覆蓋的寬高比例（0~1）")
    ap.add_argument("--outdir", default=".", help="輸出資料夾")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    root = Path(".")
    eye_csv = detect_latest_eye(root, args.stamp)
    stamp = eye_csv.stem.replace("video_eye_", "")
    print(f"[EYE] {eye_csv.name}")

    df = pd.read_csv(eye_csv, low_memory=False)

    # 1) 取得 3D 注視點
    C,R,U,N,W,H = get_screen_basis(DEFAULT_PLANE)
    if all(c in df.columns for c in ("hit_x","hit_y","hit_z")):
        P = df[["hit_x","hit_y","hit_z"]].to_numpy(float)
    elif all(c in df.columns for c in ("GazePointX","GazePointY","GazePointZ")):
        P = df[["GazePointX","GazePointY","GazePointZ"]].to_numpy(float)
    else:
        need_o = all(c in df.columns for c in
            ["LeftEye_PosX","LeftEye_PosY","LeftEye_PosZ","RightEye_PosX","RightEye_PosY","RightEye_PosZ"])
        need_d = all(c in df.columns for c in
            ["CombinedGazeForward_X","CombinedGazeForward_Y","CombinedGazeForward_Z"])
        if not (need_o and need_d):
            raise RuntimeError("缺少 hit_x/y/z 或 (眼睛位置 + CombinedGazeForward_*)")
        oL = df[["LeftEye_PosX","LeftEye_PosY","LeftEye_PosZ"]].to_numpy(float)
        oR = df[["RightEye_PosX","RightEye_PosY","RightEye_PosZ"]].to_numpy(float)
        O = 0.5*(oL+oR)
        D = df[["CombinedGazeForward_X","CombinedGazeForward_Y","CombinedGazeForward_Z"]].to_numpy(float)
        D = D/(np.linalg.norm(D, axis=1, keepdims=True)+1e-12)
        P = rays_to_plane_intersection(O,D,C,N)

    # 2) UV 與 on-screen
    u,v,d = project_uv(P,C,R,U,N,W,H)
    on = (np.abs(d)<=EPS) & (u>=0)&(u<=1)&(v>=0)&(v<=1)

    if not np.any(on):
        print("[WARN] 無 on-screen 樣本，跳過可視化。")
        return

    # ---------------- 中心五區 ----------------
    u_on = u[on]; v_on = v[on]
    lab, (cu_min, cu_max, cv_min, cv_max) = label_center5(u_on, v_on, center_frac=args.center_frac)
    names = ["Center", "Top", "Bottom", "Left", "Right"]
    counts5 = np.array([(lab==i).sum() for i in range(5)], float)
    total5 = counts5.sum()
    perc5 = (counts5/total5*100.0) if total5>0 else counts5

    out_csv = outdir / f"{stamp}__aoi_center5.csv"
    pd.DataFrame({"AOI": names, "Percent": perc5}).to_csv(out_csv, index=False)
    print(f"[OK] Center-5 CSV → {out_csv}")

    def U(u): return 16.0 * u
    def V(v): return  9.0 * v

    u0,u1,u2,u3 = 0.0, cu_min, cu_max, 1.0
    v0,v1,v2,v3 = 0.0, cv_min, cv_max, 1.0
    rects5 = [
        ("Center", (u1, v1, u2-u1, v2-v1), perc5[0]),
        ("Top",    (u0, v2, u3-u0, v3-v2), perc5[1]),
        ("Bottom", (u0, v0, u3-u0, v1-v0), perc5[2]),
        ("Left",   (u0, v1, u1-u0, v2-v1), perc5[3]),
        ("Right",  (u2, v1, u3-u2, v2-v1), perc5[4]),
    ]

    vmax5 = max(100.0, float(np.nanmax(perc5)) if np.isfinite(perc5).any() else 100.0)
    norm5 = Normalize(vmin=0.0, vmax=vmax5)
    cmap = get_cmap(None)

    fig2, ax2 = plt.subplots(figsize=(12.8, 7.2))
    ax2.set_xlim(0, 16); ax2.set_ylim(0, 9); ax2.set_aspect('equal')
    ax2.set_title(f"Center-5 AOI (filled) — {stamp}")
    for name, (ux, vy, uw, vh), val in rects5:
        x, y, w, h = U(ux), V(vy), U(uw), V(vh)
        face = cmap(norm5(val))
        ax2.add_patch(Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor=face))
        txt_color = 'black' if luminance(face) > 0.5 else 'white'
        ax2.text(x + w/2, y + h/2, f"{name}\n{val:.1f}%", ha="center", va="center", fontsize=12, color=txt_color)

    sm5 = ScalarMappable(norm=norm5, cmap=cmap); sm5.set_array([])
    fig2.colorbar(sm5, ax=ax2, fraction=0.046, pad=0.04, label="On-screen %")
    ax2.add_patch(Rectangle((0,0), 16, 9, fill=False, linewidth=2))
    fig2.tight_layout()
    out_png_center = outdir / f"{stamp}__aoi_center5.png"
    fig2.savefig(out_png_center, dpi=150); plt.close(fig2)
    print(f"[OK] Center-5 Screen Fill → {out_png_center}")

    # ---------------- Nx×Ny GridFill ----------------
    yi, xi = quantize_uv(u[on], v[on], args.nx, args.ny)
    counts = np.zeros((args.ny, args.nx), int)
    np.add.at(counts, (yi, xi), 1)
    total = counts.sum()
    perc = (counts/total*100.0) if total>0 else counts.astype(float)

    vmaxg = max(100.0, float(np.nanmax(perc)) if np.isfinite(perc).any() else 100.0)
    normg = Normalize(vmin=0.0, vmax=vmaxg)

    fig3, ax3 = plt.subplots(figsize=(12.8, 7.2))
    ax3.set_xlim(0, 16); ax3.set_ylim(0, 9); ax3.set_aspect('equal')
    ax3.set_title(f"{args.nx}×{args.ny} AOI (filled) — {stamp}")

    for y in range(args.ny):
        v_low  = y/args.ny
        v_high = (y+1)/args.ny
        for x in range(args.nx):
            u_low  = x/args.nx
            u_high = (x+1)/args.nx
            val = float(perc[y, x])
            x0, y0 = U(u_low), V(v_low)
            w,  h  = U(u_high-u_low), V(v_high-v_low)
            face = cmap(normg(val))
            ax3.add_patch(Rectangle((x0, y0), w, h, linewidth=1.0, edgecolor='black', facecolor=face))
            txt_color = 'black' if luminance(face) > 0.5 else 'white'
            ax3.text(x0 + w/2, y0 + h/2, f"{val:.1f}%", ha="center", va="center", fontsize=10, color=txt_color)

    ax3.add_patch(Rectangle((0,0), 16, 9, fill=False, linewidth=2))
    for k in range(1, args.nx):
        xg = U(k/args.nx); ax3.plot([xg, xg], [0, 9], linewidth=1.0)
    for k in range(1, args.ny):
        yg = V(k/args.ny); ax3.plot([0, 16], [yg, yg], linewidth=1.0)

    smg = ScalarMappable(norm=normg, cmap=cmap); smg.set_array([])
    fig3.colorbar(smg, ax=ax3, fraction=0.046, pad=0.04, label="On-screen %")
    fig3.tight_layout()
    out_png_grid = outdir / f"{stamp}__aoi_gridfill_{args.nx}x{args.ny}.png"
    fig3.savefig(out_png_grid, dpi=150); plt.close(fig3)
    print(f"[OK] GridFill {args.nx}x{args.ny} → {out_png_grid}")

if __name__ == "__main__":
    main()
