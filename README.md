# Attention Gaze Kit — (English)

A lightweight toolkit for eye‑tracking analysis on a **Unity‑like screen plane** in VR. It includes three scripts:

- `AttentionGazeKit.py`: **Attention windowing** (BCEA / GTE / FixRatio / Blink*, within‑session percentile calibration + hysteresis labeling).
- `AOIHeatmap.py`: **AOI occupancy visualization** (Center‑5, N×M 16:9 filled panels, matrix heatmap).
- `DataRadar.py`: **Radar charts** (overall median profile & Top‑K windows).

> **Use the same `STAMP` for a recording pair**:  
> `video_eye_20251001_211123.csv` and `video_face_20251001_211123.csv`.

---

## 1) Folder structure (current)

```
ProjectRoot/
├─ GazedataAnalysis.py           # One‑click runner (recommended)
├─ Attention Script/             # The three analysis scripts
│  ├─ AttentionGazeKit.py
│  ├─ AOIHeatmap.py
│  └─ DataRadar.py
├─ Raw Data/                     # ← Put your original CSVs here
│  ├─ video_eye_<STAMP>.csv
│  └─ video_face_<STAMP>.csv
└─ AnalysisData/                 # ← Outputs (auto‑created)
   ├─ pic/
   │  └─ <STAMP>/               # All image outputs (.png/.jpg/.jpeg/.svg)
   └─ CSV/
      └─ <STAMP>/               # All CSV and other numeric outputs (incl. <STAMP>__windows.csv)
```

> Example `<STAMP>`: `20251001_211123`.  
> Internally, the runner writes to a staging folder `AnalysisData/_staging/<STAMP>/` while running all three steps, then **moves** files into `pic/<STAMP>/` and `CSV/<STAMP>/`.

---

## 2) Requirements

Python 3.9+ recommended.

```bash
pip install numpy pandas matplotlib
# Optional: robust CSV encoding detection used by AttentionGazeKit
pip install charset-normalizer chardet
```

---

## 3) One‑click run (recommended)

Place the raw CSVs under **Raw Data/**, then execute in **ProjectRoot**:

```bash
# Auto-detect the latest video_eye_*.csv in Raw Data/
python GazedataAnalysis.py

# Or run a specific STAMP
python GazedataAnalysis.py 20251001_211123

# Common options
python GazedataAnalysis.py --data-dir "Raw Data" --nx 4 --ny 4 --center-frac 0.6 --topk 5

# Forward extra args to AttentionGazeKit.py (e.g., skip plotting)
python GazedataAnalysis.py 20251001_211123 --extra --no-plot
```

**Where outputs go**  
- Images → `AnalysisData/pic/<STAMP>/`  
- CSV / other numeric files → `AnalysisData/CSV/<STAMP>/`

---

## 4) Advanced: run scripts individually (optional)

If you prefer not to use the runner, you can call each script manually (you must manage working directories and output paths yourself):

```bash
# A) Attention windows (produces <STAMP>__windows.csv)
python "Attention Script/AttentionGazeKit.py" <STAMP> --outdir results

# B) AOI visualizations (Center‑5, N×M, heatmap)
python "Attention Script/AOIHeatmap.py" <STAMP> --nx 4 --ny 4 --center-frac 0.6 --outdir results

# C) Radar charts (requires an existing <STAMP>__windows.csv)
python "Attention Script/DataRadar.py" <STAMP> --topk 5 --outdir results
```

> **Important**: When running a single script, ensure the **current working directory** can read `video_eye_<STAMP>.csv` / `video_face_<STAMP>.csv`, and set `--outdir` to where you want the outputs.

---

## 5) Expected input columns (key points)

- **Time column (any one)**: `video_time_s`, `time_s`, `t`, `time`, `timestamp`, `sec`, `seconds`, `frame_time` (or any numeric, non‑decreasing time column).
- **Gaze data (choose one)**  
  1) Direct 3D gaze hits: `hit_x, hit_y, hit_z` (or `GazePointX, GazePointY, GazePointZ`), or  
  2) Eye origins + combined forward ray: `LeftEye_PosX/Y/Z`, `RightEye_PosX/Y/Z` + `CombinedGazeForward_X/Y/Z` (ray‑plane intersection to the screen).
- **Face data (optional)**: `EyesClosedL`/`EyesClosedR` in [0,1] for blink metrics; merged with eye stream by nearest neighbor using `T_SYNC` (seconds).

---

## 6) Model & metrics (overview)

- **UV projection**: All scripts share `DEFAULT_PLANE` (fields: `center`, unit `right/up`, `width/height`). Points are mapped to **UV ∈ [0,1]²**; “on‑screen” also checks a small plane‑distance epsilon `EPS`.
- **BCEA**: Bivariate contour ellipse area (mass `BCEA_P`). Smaller = tighter dispersion (better).
- **GTE**: First‑order gaze transition entropy over AOIs **including the off‑screen state**. Smaller = more stable targeting.
- **FixRatio (IVT)**: Fixation ratio via a simple velocity threshold in UV space (`VT`) and min duration (`TMIN`). Larger = more stable fixations.
- **BlinkRate / BlinkFrac**: Enabled when face data is present.
- **AttentionScore**: Convert metrics to **direction‑aligned percentiles** (invert smaller‑is‑better metrics), average them, then an **outer percentile** maps to [0,1].
- **Focused flag**: Hysteresis with high/low thresholds (`THR_HIGH`/`THR_LOW`) and min on/off durations (`MIN_ON`/`MIN_OFF`).

---

## 7) Troubleshooting

- **Missing eye/face CSVs**: Ensure the pair uses the same `STAMP` and is placed under `Raw Data/`.
- **No time column detected**: Provide a numeric, non‑decreasing time column or rename to a common name listed above.
- **Mostly off‑screen UV points**: Check `DEFAULT_PLANE` (center, size, axes) and `EPS` tolerance.
- **No radar output**: Run the attention step first and confirm `<STAMP>__windows.csv` exists.

---

## 8) Reproducibility tips

- Keep `DEFAULT_PLANE` and `EPS` **identical** across all scripts to avoid UV mismatches.
- If you want comparability with AOI heatmaps, use the **same AOI grid** in `AttentionGazeKit.py` and `AOIHeatmap.py` (`NX×NY` vs `--nx/--ny`).
- Output per recording under its own `<STAMP>` subfolder (already handled by the runner).

---

# Attention Gaze Kit — 使用說明

一個針對 **VR／Unity 螢幕平面** 的輕量級注視分析工具組，包含三個腳本：

- `AttentionGazeKit.py`：**注意力視窗計算**（BCEA／GTE／FixRatio／Blink*，場內百分位校準 + Hysteresis 標註）。
- `AOIHeatmap.py`：**AOI 佔用視覺化**（中心五區、N×M 16:9 填色、矩陣熱圖）。
- `DataRadar.py`：**雷達圖**（整體中位數與 Top-K 視窗）。

> **同批資料以相同 `STAMP` 命名**：  
> `video_eye_20251001_211123.csv`、`video_face_20251001_211123.csv`。

---

## 1) 資料夾結構（新版）

```
專案根目錄/
├─ GazedataAnalysis.py           # 一鍵執行的 runner（建議用它）
├─ Attention Script/             # 三個分析腳本
│  ├─ AttentionGazeKit.py
│  ├─ AOIHeatmap.py
│  └─ DataRadar.py
├─ Raw Data/                     # ← 原始 CSV 放這裡
│  ├─ video_eye_<STAMP>.csv
│  └─ video_face_<STAMP>.csv
└─ AnalysisData/                 # ← 分類輸出（自動建立）
   ├─ pic/
   │  └─ <STAMP>/               # 所有圖片輸出（.png/.jpg/.jpeg/.svg）
   └─ CSV/
      └─ <STAMP>/               # 所有 CSV 與其他數據檔（含 <STAMP>__windows.csv）
```

> `<STAMP>` 例：`20251001_211123`。  
> 內部會用暫存夾 `AnalysisData/_staging/<STAMP>/` 完成三步分析後，再自動把檔案分流到 `pic/<STAMP>/` 與 `CSV/<STAMP>/`。

---

## 2) 安裝需求

Python 3.9+ 建議。

```bash
pip install numpy pandas matplotlib
# 可選：自動偵測 CSV 編碼（AttentionGazeKit 使用）
pip install charset-normalizer chardet
```

---

## 3) 一鍵執行（建議）

把原始資料放到 **Raw Data/**，於「專案根目錄」執行：

```bash
# 自動抓 Raw Data/ 內最新的 video_eye_*.csv
python GazedataAnalysis.py

# 指定某個 STAMP
python GazedataAnalysis.py 20251001_211123

# 常用參數
python GazedataAnalysis.py --data-dir "Raw Data" --nx 4 --ny 4 --center-frac 0.6 --topk 5

# 傳遞額外參數給 AttentionGazeKit.py（例：不輸出注意力圖）
python GazedataAnalysis.py 20251001_211123 --extra --no-plot
```

**輸出位置**  
- 圖片 → `AnalysisData/pic/<STAMP>/`  
- CSV／其他數據 → `AnalysisData/CSV/<STAMP>/`

---

## 4) 進階：單支腳本（可選）

若你不使用 runner，也可各自執行（需自行處理工作目錄與輸出夾）：

```bash
# A) 注意力（會產生 <STAMP>__windows.csv）
python "Attention Script/AttentionGazeKit.py" <STAMP> --outdir results

# B) AOI 視覺化（中心五區、N×M、熱圖）
python "Attention Script/AOIHeatmap.py" <STAMP> --nx 4 --ny 4 --center-frac 0.6 --outdir results

# C) 雷達圖（需已存在 <STAMP>__windows.csv）
python "Attention Script/DataRadar.py" <STAMP> --topk 5 --outdir results
```

> **注意**：若直接跑單支腳本，請確保目前工作目錄能讀到 `video_eye_<STAMP>.csv`／`video_face_<STAMP>.csv`，且 `--outdir` 指向你要的輸出夾。

---

## 5) 輸入資料欄位（重點）

- **時間欄（其一即可）**：`video_time_s`、`time_s`、`t`、`time`、`timestamp`、`sec`、`seconds`、`frame_time`（或任何數值且單調不減欄位）。
- **注視資料（擇一）**  
  1) 直接提供 3D 注視點：`hit_x, hit_y, hit_z`（或 `GazePointX, GazePointY, GazePointZ`）  
  2) 提供眼睛原點與合成視線：`LeftEye_PosX/Y/Z`、`RightEye_PosX/Y/Z` + `CombinedGazeForward_X/Y/Z`（用射線與螢幕平面求交）
- **臉部資料（選用）**：`EyesClosedL`／`EyesClosedR`（0~1），用於 Blink 指標；與 Eye 以 `T_SYNC`（秒）做最近對齊。

---

## 6) 模型與指標（概要）

- **UV 投影**：三個腳本共用 `DEFAULT_PLANE`（`center`、單位 `right/up`、`width/height`），UV ∈ [0,1]² 視為螢幕內（距離平面 < `EPS`）。  
- **BCEA**：注視分散橢圓面積（質量 `BCEA_P`）。小＝集中。  
- **GTE**：AOI（含 off-screen）的一階轉移熵。小＝穩定。  
- **FixRatio（IVT）**：UV 速度閾值 `VT`、最短凝視 `TMIN`；比例大＝穩定注視多。  
- **BlinkRate／BlinkFrac**：有臉部資料才啟用。  
- **AttentionScore**：各指標轉為場內百分位、方向一致化後等權平均，外層百分位再校準至 \[0,1\]。  
- **Focused**：雙門檻（`THR_HIGH/LOW`）＋最短 on/off 持續（`MIN_ON/OFF`）。

---

## 7) 疑難排解

- **找不到 eye/face CSV**：確認兩檔名同 `STAMP` 且放在 `Raw Data/`。  
- **讀不到時間欄**：提供一個數值且單調不減的時間欄位或改欄名為常見名稱。  
- **UV 點多為螢幕外**：檢查 `DEFAULT_PLANE` 之中心、尺寸、向量，以及 `EPS` 容差。  
- **雷達圖沒產生**：需先跑注意力，確保 `<STAMP>__windows.csv` 存在。

---

## 8) 再現性建議

- **務必在三個腳本維持相同 `DEFAULT_PLANE` 與 `EPS`**，避免 UV 不一致。  
- 若要與 AOI 熱圖對比，建議 `AttentionGazeKit.py` 的 `NX×NY` 與 `AOIHeatmap.py` 的 `--nx/--ny` 使用相同切格。  
- 每次錄製可用 `<STAMP>` 做輸出子夾，便於追蹤與比對（runner 已自動處理）。
