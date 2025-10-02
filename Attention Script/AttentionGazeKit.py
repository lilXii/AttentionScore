#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screen-AOI Attention (PR-calibrated)

- 成對讀取：video_eye_*.csv + video_face_*.csv（可省略 STAMP 自動抓最新）
- 以 eye 為主時間軸，face merge_asof 近鄰對齊（容差 CONFIG['T_SYNC'] 秒）
- 固定 Unity 螢幕平面 → 3D 投影到 UV ∈ [0,1]^2；螢幕外(off-screen)獨立 state（供 GTE）
- 指標：BCEA(螢幕內)、GTE(AOI含off)、FixRatio(IVT, UV)、BlinkRate/BlinkFrac(可選)
- 合成：per-metric 場內百分位(方向統一) 等權平均 → AttentionRaw(0..1)
       再做一次「外層百分位校準」→ AttentionScore(0..1, PR)
       用 PR 分數進行 Hysteresis（二門檻+最短持續）得到 Focused

極簡用法：
  python AttentionGazeKit.py                         # 自動找最新一組
  python AttentionGazeKit.py 20251001_211123         # 指定 STAMP
  python AttentionGazeKit.py 20251001_211123 --outdir results --no-plot
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd

# 繪圖（輸出檔案用非互動後端）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 日誌 與 編碼偵測 ----------
import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

try:
    from charset_normalizer import from_path as _cn_from_path  # 推薦
    _HAS_CN = True
except Exception:
    _HAS_CN = False
try:
    import chardet  # 備用
    _HAS_CHARDET = True
except Exception:
    _HAS_CHARDET = False


# ---------- 固定設定（想調整直接改這裡） ----------
CONFIG = dict(
    NX=4, NY=4,               # AOI 切格
    EPS=0.05,                 # 螢幕平面距離容差（世界座標）
    T_SYNC=0.02,              # eye/face 合併容差（秒）
    BCEA_P=0.682,             # BCEA 質量
    WIN=5.0, STEP=1.0,        # 滑動窗長度/步長（秒）
    MIN_VALID_FRAC=0.30,      # 每窗螢幕內有效點比例下限
    VT=60.0, TMIN=0.1,        # IVT 凝視參數（UV空間）
    USE_FIXRATIO=True,
    USE_BLINK=True,           # 若 face 有 EyesClosed* 就會啟用
    BLINK_TH=0.5, BLINK_MIN=0.05, BLINK_MAX=0.40, BLINK_MERGE=0.08,
    # 注意力 → 專注 標籤：雙門檻 + 最短持續（用外層 PR 分數）
    THR_HIGH=0.75, THR_LOW=0.65, MIN_ON=3, MIN_OFF=2,
)

# ---------- 固定 Unity 螢幕平面 ----------
DEFAULT_PLANE = {
    "center": {"x": 0.33000001311302187, "y": 1.2999999523162842, "z": 4.929999828338623},
    "right":  {"x": 1.0, "y": 0.0, "z": 0.0},
    "up":     {"x": 0.0, "y": 1.0, "z": 0.0},
    "width":  7.950000286102295,
    "height": 4.471874713897705,
}

def _v(d: Dict[str, float]) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"]], dtype=float)

def get_screen_basis(pdct=DEFAULT_PLANE):
    C = _v(pdct["center"])
    R = _v(pdct["right"]); R /= (np.linalg.norm(R) + 1e-12)
    U = _v(pdct["up"]);    U /= (np.linalg.norm(U) + 1e-12)
    N = np.cross(R, U);    N /= (np.linalg.norm(N) + 1e-12)
    W = float(pdct["width"]); H = float(pdct["height"])
    return C, R, U, N, W, H


# ---------- 讀檔/時間欄 ----------
def read_csv_fallback(p: Path) -> pd.DataFrame:
    """
    先嘗試自動偵測編碼（charset-normalizer -> chardet），失敗再少量回退。
    任何失敗都會回報哪個編碼與錯誤內容，方便追查。
    """
    # 1) 嘗試偵測
    if _HAS_CN:
        try:
            res = _cn_from_path(str(p)).best()
            if res is not None and res.encoding:
                enc = res.encoding
                logger.info(f"read_csv: charset-normalizer detected encoding={enc}")
                return pd.read_csv(p, encoding=enc, low_memory=False)
        except Exception as e:
            logger.warning(f"charset-normalizer failed: {e}")
    elif _HAS_CHARDET:
        try:
            with open(p, "rb") as f:
                raw = f.read(1024 * 1024)  # 讀前 1MB 判斷
            det = chardet.detect(raw)
            enc = det.get("encoding", None)
            if enc:
                logger.info(f"read_csv: chardet detected encoding={enc}")
                return pd.read_csv(p, encoding=enc, low_memory=False)
        except Exception as e:
            logger.warning(f"chardet failed: {e}")

    # 2) 小清單回退
    tried = []
    for enc in ["utf-8", "utf-8-sig", "cp950", "big5", "latin-1"]:
        try:
            logger.info(f"read_csv: try encoding={enc}")
            return pd.read_csv(p, encoding=enc, low_memory=False)
        except Exception as e:
            tried.append(f"{enc}: {e}")

    # 3) 宣告失敗
    raise RuntimeError(f"Failed to read CSV {p}. Tried encodings:\n" + "\n".join(tried))

def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    cols_lc = {c.lower(): c for c in df.columns}
    # 白名單
    for k in ["video_time_s", "time_s", "t", "time", "timestamp", "sec", "seconds", "frame_time"]:
        if k in cols_lc:
            return cols_lc[k]
    # 啟發式：數值且單調不減的第一欄
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().all() and v.is_monotonic_increasing:
            logger.info(f"detect_time_col: heuristically selected '{c}'")
            return c
    return None

def to_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy()


# ---------- 檔名配對 ----------
def find_eye_face_pair(root: Path, stamp: Optional[str]) -> Tuple[Path, Path]:
    eyes = sorted(root.glob("video_eye_*.csv"))
    faces = sorted(root.glob("video_face_*.csv"))
    if not eyes or not faces:
        raise FileNotFoundError("缺少 video_eye_*.csv 或 video_face_*.csv")

    def suf(p: Path, prefix: str) -> str:
        name = p.name; assert name.startswith(prefix)
        return name[len(prefix):-4]

    em = {suf(p,"video_eye_"): p for p in eyes}
    fm = {suf(p,"video_face_"): p for p in faces}
    common = sorted(set(em) & set(fm))
    if not common:
        raise FileNotFoundError("找不到共同時間戳的 eye/face 成對檔案")

    key = stamp if stamp else common[-1]
    if key not in common:
        raise FileNotFoundError(f"指定 stamp={stamp} 不存在。")
    return em[key], fm[key]


# ---------- 幾何 ----------
def rays_to_plane_intersection(O: np.ndarray, D: np.ndarray, C: np.ndarray, N: np.ndarray) -> np.ndarray:
    # t = ((C-O)·N) / (D·N)；顯式過濾 denom≈0，結果設 NaN
    denom = (D @ N)
    mask = np.abs(denom) > 1e-12
    t = np.full(O.shape[0], np.nan, dtype=float)
    if mask.any():
        t[mask] = ((C - O[mask]) @ N) / denom[mask]
        t[t < 0] = np.nan
    P = O + D * t[:, None]
    return P

def project_uv_from_points(P: np.ndarray, C: np.ndarray, R: np.ndarray, U: np.ndarray, N: np.ndarray,
                           W: float, H: float):
    if not np.isfinite(W) or not np.isfinite(H) or (W <= 1e-9) or (H <= 1e-9):
        raise ValueError(f"Invalid screen size W={W}, H={H}")
    p = P - C[None, :]
    xloc = p @ R
    yloc = p @ U
    dplane = p @ N
    u = xloc / W + 0.5
    v = yloc / H + 0.5
    return u, v, dplane


# ---------- AOI/指標 ----------
def quantize_aoi_uv(u, v, nx, ny):
    xi = np.clip(np.floor(u * nx).astype(int), 0, nx-1)
    yi = np.clip(np.floor(v * ny).astype(int), 0, ny-1)
    return yi * nx + xi

def bcea(x: np.ndarray, y: np.ndarray, p: float = CONFIG["BCEA_P"]) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]; y = y[valid]
    if x.size < 3:
        logger.warning("BCEA: valid samples < 3; return NaN")
        return np.nan
    sx = np.std(x); sy = np.std(y)
    if (sx < 1e-12) or (sy < 1e-12):
        return 0.0
    try:
        rho = np.corrcoef(x, y)[0, 1] if x.size > 1 else 0.0
    except Exception:
        rho = 0.0
    rho = float(np.clip(rho, -0.999999, 0.999999))
    k = -np.log(max(1.0 - p, 1e-9))
    return float(2.0 * np.pi * k * sx * sy * np.sqrt(1.0 - rho * rho))

def mark_fixations_ivt(t, x, y, on, v_th=CONFIG["VT"], t_min=CONFIG["TMIN"]):
    t = np.asarray(t, float); x = np.asarray(x, float); y = np.asarray(y, float); on = np.asarray(on, bool)
    is_fix = np.zeros_like(t, dtype=bool)
    if len(t) < 3: return is_fix
    dt = np.diff(t); dt = np.append(dt, dt[-1] if dt.size else 0.0); dt[dt<=0]=np.nan
    dx = np.diff(x, prepend=x[0]); dy = np.diff(y, prepend=y[0])
    speed = np.sqrt(dx*dx + dy*dy) / dt; speed[~np.isfinite(speed)] = np.nan
    cand = (speed <= v_th) & on
    i=0; n=len(t)
    while i<n:
        if not cand[i]: i+=1; continue
        j=i+1
        while j<n and cand[j]: j+=1
        dur = (t[j-1]-t[i]) if j-1>i else 0.0
        if dur >= t_min: is_fix[i:j]=True
        i=j
    return is_fix

def gte_from_sequence(seq, K=None, eps=1e-9):
    s = np.asarray(seq, int)
    if s.size < 3: return np.nan
    if K is None: K = int(np.nanmax(s))+1
    K = max(int(K),1)
    T = np.zeros((K,K), float)
    prev, nxt = s[:-1], s[1:]
    for i,j in zip(prev, nxt):
        if 0<=i<K and 0<=j<K: T[i,j]+=1.0
    rowsum = T.sum(axis=1, keepdims=True) + eps
    P = T/rowsum
    pi = rowsum[:,0]
    if pi.sum()<=0: return np.nan
    pi = pi / pi.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        logP = np.log2(np.clip(P, eps, 1.0))
    H_rows = -np.nansum(P*logP, axis=1)
    return float(np.nansum(pi*H_rows))


# ---------- BLINK ----------
def extract_closed_prob(face_df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> Optional[np.ndarray]:
    pick = [c for c in (cols or ["EyesClosedL","EyesClosedR"]) if c in face_df.columns]
    if not pick: return None
    mat = np.vstack([pd.to_numeric(face_df[c], errors="coerce").to_numpy() for c in pick])
    closed = np.nanmax(mat, axis=0)
    return np.clip(closed.astype(float), 0.0, 1.0)

def detect_blinks(t, c, th=CONFIG["BLINK_TH"], min_d=CONFIG["BLINK_MIN"],
                  max_d=CONFIG["BLINK_MAX"], merge_gap=CONFIG["BLINK_MERGE"]):
    t = np.asarray(t, float); c = np.asarray(c, float)
    ok = np.isfinite(t) & np.isfinite(c)
    if ok.sum() < 2: return 0, 0.0
    t = t[ok]; c = c[ok]
    mask = (c >= th)
    runs=[]; i=0; n=len(mask)
    while i<n:
        v=mask[i]; j=i+1
        while j<n and mask[j]==v: j+=1
        runs.append((i,j,v)); i=j
    merged=[]; k=0
    while k<len(runs):
        i0,j0,v0=runs[k]
        if v0 and k+2<len(runs):
            i1,j1,v1=runs[k+1]; i2,j2,v2=runs[k+2]
            gap=(t[j1-1]-t[i1]) if j1-1>i1 else 0.0
            if (not v1) and (gap<=merge_gap) and v2:
                merged.append((i0,j2,True)); k+=3; continue
        merged.append((i0,j0,v0)); k+=1
    count=0; closed_time=0.0
    for a,b,v in merged:
        if v:
            dur=(t[b-1]-t[a]) if b-1>a else 0.0
            closed_time+=dur
            if (dur>=min_d) and (dur<=max_d): count+=1
    return count, closed_time


# ---------- 百分位 & 去抖動 ----------
def percentile_rank_vector(v: np.ndarray) -> np.ndarray:
    """
    場內百分位（mid-rank，分母用 N+1 防 0/1 飽和）
    - 全 NaN -> 全 NaN
    - 全常數（非 NaN）-> 全 0.5
    """
    v = np.asarray(v, float)
    p = np.full_like(v, np.nan, dtype=float)

    finite = np.isfinite(v)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return p

    x = v[idx]
    if np.unique(x).size == 1:
        p[idx] = 0.5
        return p

    order = np.argsort(x, kind="mergesort")
    ranks_srt = np.arange(1, len(x) + 1, dtype=float)

    # tie → 排序區間 [i:j) 設為平均名次
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks_srt[i:j] = avg
        i = j

    ranks = np.empty_like(ranks_srt)
    ranks[order] = ranks_srt

    p[idx] = ranks / (len(x) + 1.0)
    return p

def combine_attention_percentile(win_df: pd.DataFrame,
                                 use_fixratio: bool = True,
                                 use_blink: bool = True) -> np.ndarray:
    """
    將各指標轉成場內百分位後等權平均（0..1）：
      - 小越好：BCEA, GTE, BlinkRate, BlinkFrac → 用 (1 - p)
      - 大越好：FixRatio → 用 p
    缺失值：只對有值的成分做平均。
    """
    parts: List[np.ndarray] = []
    # smaller is better
    p_b = percentile_rank_vector(win_df["BCEA"].to_numpy(dtype=float)); parts.append(1.0 - p_b)
    p_g = percentile_rank_vector(win_df["GTE"].to_numpy(dtype=float));  parts.append(1.0 - p_g)
    # larger is better
    if use_fixratio and "FixRatio" in win_df.columns:
        p_f = percentile_rank_vector(win_df["FixRatio"].to_numpy(dtype=float)); parts.append(p_f)
    if use_blink:
        if "BlinkRate" in win_df.columns:
            p_br = percentile_rank_vector(win_df["BlinkRate"].to_numpy(dtype=float)); parts.append(1.0 - p_br)
        if "BlinkFrac" in win_df.columns:
            p_bf = percentile_rank_vector(win_df["BlinkFrac"].to_numpy(dtype=float)); parts.append(1.0 - p_bf)
    valid = [a for a in parts if a is not None and np.isfinite(a).any()]
    if not valid: return np.full(len(win_df), np.nan)
    return np.nanmean(np.vstack(valid), axis=0)

def hysteresis_mask(score: np.ndarray, thr_high: float, thr_low: float,
                    min_on: int = 1, min_off: int = 1) -> np.ndarray:
    """
    NaN 不改變狀態且不重置計數；雙門檻 + 最短持續。
    """
    s = np.asarray(score, float)
    n = len(s)
    on = np.zeros(n, dtype=bool)
    state = False
    st_on = 0
    st_off = 0
    for i, v in enumerate(s):
        if not np.isfinite(v):
            on[i] = state
            continue
        if not state:
            if v >= thr_high:
                st_on += 1
                if st_on >= min_on:
                    state = True
                    st_off = 0
            else:
                st_on = 0
        else:
            if v <= thr_low:
                st_off += 1
                if st_off >= min_off:
                    state = False
                    st_on = 0
            else:
                st_off = 0
        on[i] = state
    return on


# ---------- 視窗索引器（效能小優化） ----------
def iter_windows_indices(t: np.ndarray, win: float, step: float):
    """
    根據時間軸 t（已排序）回傳一系列 (i0, i1, s, e) 的窗索引。
    """
    t = np.asarray(t, float)
    out = []
    if t.size == 0:
        return out
    i0 = 0
    while True:
        s = t[i0]
        e = s + win
        i1 = np.searchsorted(t, e, side="left")
        if i1 - i0 >= 3:
            out.append((i0, i1, s, e))
        s_next = s + step
        if s_next + win > t[-1] + 1e-9:
            break
        i0 = np.searchsorted(t, s_next, side="left")
        if i0 >= len(t) - 1:
            break
    return out


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stamp", nargs="?", default=None, help="時間戳（如 20251001_211123），省略為最新一組")
    ap.add_argument("--outdir", default=".", help="輸出資料夾")
    ap.add_argument("--no-plot", action="store_true", help="不輸出圖檔")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    root = Path(".")
    eye_path, face_path = find_eye_face_pair(root, args.stamp)
    print(f"[PAIR] eye = {eye_path.name}")
    print(f"[PAIR] face = {face_path.name}")

    # 讀檔
    eye_df = read_csv_fallback(eye_path)
    face_df = read_csv_fallback(face_path)

    # 時間欄
    col_te = detect_time_col(eye_df)
    col_tf = detect_time_col(face_df)
    if col_te is None or col_tf is None:
        raise RuntimeError("找不到時間欄位（video_time_s / time_s / t / time / timestamp / sec ...）")

    eye_df = eye_df.sort_values(col_te).reset_index(drop=True)
    face_df = face_df.sort_values(col_tf).reset_index(drop=True)
    eye_df["t_eye"] = to_numeric(eye_df[col_te]).astype(float)
    face_df["t_face"] = to_numeric(face_df[col_tf]).astype(float)

    # BLINK 對齊
    face_closed = extract_closed_prob(face_df) if CONFIG["USE_BLINK"] else None
    face_keep = face_df[["t_face"]].copy()
    if face_closed is not None:
        face_keep["closed_prob"] = face_closed
    merged = pd.merge_asof(
        eye_df.sort_values("t_eye"),
        face_keep.sort_values("t_face"),
        left_on="t_eye", right_on="t_face",
        direction="nearest", tolerance=CONFIG["T_SYNC"]
    )
    merged["face_synced"] = np.isfinite(merged.get("t_face", np.nan))

    # 投影 UV
    C,R,U,N,W,H = get_screen_basis(DEFAULT_PLANE)
    # 先試現成 3D gaze hit
    P = None
    for cand in [("hit_x","hit_y","hit_z"), ("GazePointX","GazePointY","GazePointZ")]:
        if all(c in eye_df.columns for c in cand):
            P = eye_df[list(cand)].to_numpy(float); break
    if P is None:
        need_o = all(c in eye_df.columns for c in [
            "LeftEye_PosX","LeftEye_PosY","LeftEye_PosZ","RightEye_PosX","RightEye_PosY","RightEye_PosZ"])
        need_d = all(c in eye_df.columns for c in [
            "CombinedGazeForward_X","CombinedGazeForward_Y","CombinedGazeForward_Z"])
        if not (need_o and need_d):
            raise RuntimeError("缺少 3D 注視點，且無 (眼睛位置 + CombinedGazeForward_*) 無法投影 UV。")
        oL = eye_df[["LeftEye_PosX","LeftEye_PosY","LeftEye_PosZ"]].to_numpy(float)
        oR = eye_df[["RightEye_PosX","RightEye_PosY","RightEye_PosZ"]].to_numpy(float)
        O = 0.5*(oL+oR)
        D = eye_df[["CombinedGazeForward_X","CombinedGazeForward_Y","CombinedGazeForward_Z"]].to_numpy(float)
        D = D/(np.linalg.norm(D, axis=1, keepdims=True)+1e-12)
        P = rays_to_plane_intersection(O,D,C,N)

    u,v,dplane = project_uv_from_points(P,C,R,U,N,W,H)
    on_screen = (np.abs(dplane)<=CONFIG["EPS"]) & (u>=0)&(u<=1)&(v>=0)&(v<=1)

    work = pd.DataFrame({
        "t": eye_df["t_eye"].to_numpy(float),
        "x": u.astype(float), "y": v.astype(float),
        "on": on_screen.astype(bool), "on_screen": on_screen.astype(bool),
        "face_synced": merged["face_synced"].astype(bool)
    })
    if "closed_prob" in merged.columns:
        work["closed_prob"] = merged["closed_prob"].astype(float)

    # AOI 與 off-screen
    aoi_base = quantize_aoi_uv(work["x"].to_numpy(), work["y"].to_numpy(), CONFIG["NX"], CONFIG["NY"])
    off_idx = CONFIG["NX"]*CONFIG["NY"]
    aoi_full = aoi_base.copy(); aoi_full[~work["on_screen"].to_numpy()] = off_idx
    work["aoi"] = aoi_full

    # Fixation（UV 空間）
    work["is_fix"] = mark_fixations_ivt(
        work["t"].values, work["x"].values, work["y"].values, work["on"].values,
        v_th=CONFIG["VT"], t_min=CONFIG["TMIN"]
    )

    # 視窗計算（用索引器）
    rows = []
    tarr = work["t"].to_numpy(float)
    for i0, i1, s, e in iter_windows_indices(tarr, CONFIG["WIN"], CONFIG["STEP"]):
        w = work.iloc[i0:i1]
        vmask = w["on"].to_numpy(bool)
        if vmask.sum() < max(5, CONFIG["MIN_VALID_FRAC"] * len(w)):
            continue

        # BCEA（螢幕內）
        b = bcea(w.loc[vmask,"x"].values, w.loc[vmask,"y"].values) if vmask.any() else np.nan

        # GTE（AOI 去重，含 off-screen）
        seq = w["aoi"].values
        if seq.size:
            seq_dedup = seq[np.insert(seq[1:]!=seq[:-1], 0, True)]
            g = gte_from_sequence(seq_dedup, K=off_idx+1, eps=1e-9)
        else:
            g = np.nan

        # FixRatio
        if CONFIG["USE_FIXRATIO"]:
            tt = w["t"].values; dts = np.diff(tt)
            if dts.size==0: fr=np.nan
            else:
                dts=np.append(dts,dts[-1]); dts[dts<=0]=np.nan
                Tfix = np.nansum(dts * w["is_fix"].values.astype(float))
                fr = Tfix / (e - s) if (e - s)>0 else np.nan
        else:
            fr=np.nan

        # BLINK（需螢幕內 + face 同步）
        if CONFIG["USE_BLINK"] and ("closed_prob" in w.columns) and vmask.any():
            tw = w.loc[vmask, "t"].values
            cw = w.loc[vmask, "closed_prob"].values
            if "face_synced" in w.columns:
                sm = w.loc[vmask, "face_synced"].values
                tw = tw[sm]; cw = cw[sm]
            cnt, ctime = detect_blinks(tw, cw)
            blink_rate = cnt/(e-s) if (e-s)>0 else np.nan
            blink_frac = ctime/(e-s) if (e-s)>0 else np.nan
        else:
            blink_rate, blink_frac = np.nan, np.nan

        rows.append((s,e,b,g,fr,blink_rate,blink_frac,len(w),int(vmask.sum())))

    win_df = pd.DataFrame(rows, columns=[
        "start","end","BCEA","GTE","FixRatio","BlinkRate","BlinkFrac","n","n_valid"
    ])

    # 合成：per-metric 百分位 → 等權平均 = AttentionRaw；外層 PR 校準 = AttentionScore
    if win_df.empty:
        win_df["AttentionRaw"] = np.nan
        win_df["AttentionScore"] = np.nan
    else:
        att_raw = combine_attention_percentile(
            win_df, use_fixratio=CONFIG["USE_FIXRATIO"], use_blink=CONFIG["USE_BLINK"]
        )
        win_df["AttentionRaw"] = att_raw
        att_pr = percentile_rank_vector(att_raw)
        win_df["AttentionScore"] = att_pr

        if np.isfinite(att_pr).any():
            win_df["Focused"] = hysteresis_mask(
                att_pr,
                thr_high=CONFIG["THR_HIGH"],
                thr_low=CONFIG["THR_LOW"],
                min_on=CONFIG["MIN_ON"],
                min_off=CONFIG["MIN_OFF"]
            ).astype(bool)
            focus_pct = float(np.nanmean(win_df["Focused"].to_numpy(dtype=bool))) * 100.0
            logger.info(f"FocusPct(overall) = {focus_pct:.1f}%  "
                        f"(thr_high={CONFIG['THR_HIGH']}, thr_low={CONFIG['THR_LOW']}, "
                        f"min_on={CONFIG['MIN_ON']}, min_off={CONFIG['MIN_OFF']})")

    # 輸出
    stamp = Path(eye_path).stem.replace("video_eye_", "")
    out_csv = Path(args.outdir) / f"{stamp}__windows.csv"
    win_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")

    if (not args.no_plot) and (not win_df.empty):
        out_png = Path(args.outdir) / f"{stamp}__attention.png"
        ctr = 0.5*(win_df["start"].to_numpy()+win_df["end"].to_numpy())
        y = win_df["AttentionScore"].to_numpy(float)  # 外層 PR
        fig = plt.figure(figsize=(12,4.2)); ax = fig.add_subplot(111)
        ax.plot(ctr, y, linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Attention (percentile, 0..1)")
        ax.set_ylim(0,1)
        ax.axhline(CONFIG["THR_HIGH"], linestyle="--", linewidth=1)
        ax.axhline(CONFIG["THR_LOW"],  linestyle="--", linewidth=1)
        ax.set_title(f"Attention (PR-calibrated) — {stamp}")
        ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
        print(f"[OK] Plot:  {out_png}")

    if not win_df.empty:
        view = win_df.sort_values("AttentionScore", ascending=False).head(10)
        print("\nTop windows by AttentionScore (PR):")
        for _, r in view.iterrows():
            print(f"  {r['start']:8.3f}-{r['end']:8.3f}s | "
                  f"AttPR={r['AttentionScore']:+6.3f} | "
                  f"BCEA={r['BCEA']:.3f}  GTE={r['GTE']:.3f}  "
                  f"Fix={r['FixRatio']:.3f}  BlinkRate={r['BlinkRate']:.3f}  BlinkFrac={r['BlinkFrac']:.3f}")

if __name__ == "__main__":
    main()
