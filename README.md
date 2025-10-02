# Attention Gaze Kit —— 使用說明（中文 README）

一個針對 **VR／Unity 螢幕平面** 的輕量級注視分析工具組，包含三個腳本：

- `AttentionGazeKit.py`：**注意力（Attention）視窗計算**（含 BCEA／GTE／FixRatio／Blink*，PR 百分位校準與 Hysteresis 專注標註）。
- `AOIHeatmap.py`：**AOI 佔用熱圖 / 中心五區 / N×M 16:9 填色圖**（僅統計螢幕內樣本）。
- `DataRadar.py`：**雷達圖**（各指標的場內百分位；整體中位數＋Top‑K 視窗）。

> **資料命名規則**：同一批資料用同一個 `STAMP`（時間戳）。例如：  
> `video_eye_20251001_211123.csv`、`video_face_20251001_211123.csv`。  
> 省略 `STAMP` 時，腳本會自動選取資料夾中**最新**的一組／一個檔案。


---

## 環境安裝

需要 Python 3.9+。

```bash
pip install numpy pandas matplotlib
#（可選）注意力腳本會嘗試自動偵測 CSV 編碼：
pip install charset-normalizer chardet
```

> 手繪圖皆使用 **非互動式 Matplotlib 後端**，可在 headless 環境直接輸出 PNG。


---

## 輸入資料格式

### 1) Eye 檔（必備）：`video_eye_<STAMP>.csv`
**時間欄**（自動偵測以下其一）：`video_time_s`、`time_s`、`t`、`time`、`timestamp`、`sec`、`seconds`、`frame_time`。若都沒有，腳本會以「數值且單調不減」的欄位嘗試。

**空間資訊（擇一）：**
- **直接提供 3D 注視點**：`hit_x, hit_y, hit_z`（或 `GazePointX, GazePointY, GazePointZ`）。
- **或提供眼睛原點與合成視線**：
  - 原點：`LeftEye_PosX/Y/Z`、`RightEye_PosX/Y/Z`
  - 合成視線方向：`CombinedGazeForward_X/Y/Z`  
  腳本會以射線與螢幕平面求交獲得 3D 注視點。

### 2) Face 檔（選用）：`video_face_<STAMP>.csv`
- **時間欄**與上方相同規則。  
- 眨眼特徵（若存在則啟用 Blink 指標）：`EyesClosedL`、`EyesClosedR`（0~1 機率或分數）。  
- 與 Eye 以最鄰近時間「**合併容差** `T_SYNC`（秒）」對齊。

> **注意**：若 Face 檔缺少 `EyesClosed*`，會自動關閉 Blink 指標，僅用 Eye 計算。


---

## 共用座標系與螢幕平面

三個腳本共用 `DEFAULT_PLANE`：
- `center`、單位化的 `right` / `up` 軸，以及實際 `width` / `height`（世界座標）。
- 將 3D 注視點投影為 **UV ∈ [0,1]²**。U、V 在 0~1 範圍內且點至平面距離 < `EPS` 視為 **螢幕內**。

> 請在三個腳本中維持相同的 `DEFAULT_PLANE` 與 `EPS`，以避免 UV 對不上。


---

## 快速開始（典型流程）

1. **計算注意力視窗**（輸出 `__windows.csv` 與注意力折線圖）：
   ```bash
   python AttentionGazeKit.py 20251001_211123 --outdir results
   ```
   （省略 `STAMP` 則自動抓最新成對 `video_eye_*` / `video_face_*`）

2. **產生 AOI 視覺化**（中心五區 + N×M 16:9 填色 + 矩陣熱圖）：
   ```bash
   python AOIHeatmap.py 20251001_211123 --nx 4 --ny 4 --center-frac 0.6 --outdir results
   ```
   （省略 `STAMP` 則自動抓最新 `video_eye_*`）

3. **繪製雷達圖**（整體中位數 + Top‑K 視窗）：
   ```bash
   python DataRadar.py 20251001_211123 --topk 5 --outdir results
   ```
   （需要步驟 1 產生的 `<STAMP>__windows.csv`）


---

## 各腳本用法與輸出

### A. `AttentionGazeKit.py` — 注意力（PR 校準 + Hysteresis）

**用法**
```bash
python AttentionGazeKit.py [STAMP] [--outdir DIR] [--no-plot]
```
- `STAMP`：可省略（自動找最新一組 eye/face）。
- `--outdir`：輸出資料夾（預設 `.`）。
- `--no-plot`：不輸出注意力圖。

**核心流程**
1. 讀取 `video_eye_<STAMP>.csv` 與 `video_face_<STAMP>.csv`（最鄰近合併，容差 `T_SYNC` 秒）。
2. 以 `DEFAULT_PLANE` 投影到 UV；螢幕外樣本標為 off‑screen（供 GTE 使用）。
3. 以**滑動窗**（長 `WIN`、步 `STEP`）計算每窗指標：
   - **BCEA**（螢幕內點的分散橢圓面積，質量 `BCEA_P`）。
   - **GTE**（AOI 狀態轉移熵；AOI 含 off‑screen）。
   - **FixRatio**（IVT：UV 速度閾值 `VT`、最短凝視 `TMIN`）。
   - **BlinkRate / BlinkFrac**（若 `EyesClosed*` 存在）。
4. 將各指標轉為**場內百分位**並方向統一（小越好取 `1-p`，大越好取 `p`），等權平均為 `AttentionRaw`。  
5. 對 `AttentionRaw` 再做一次 **外層百分位校準** 得 `AttentionScore ∈ [0,1]`。  
6. 以**雙門檻 + 最短持續**（`THR_HIGH`、`THR_LOW`、`MIN_ON`、`MIN_OFF`）產生 **Focused**（專注）片段。

**輸出**
- `<STAMP>__windows.csv`：每窗 `start,end,BCEA,GTE,FixRatio,BlinkRate,BlinkFrac,n,n_valid,AttentionRaw,AttentionScore,Focused`。
- `<STAMP>__attention.png`：`AttentionScore` 時序圖（含雙門檻線），除非 `--no-plot`。

**重要參數（可在檔頭 `CONFIG = dict(...)` 直接修改）**
- AOI 網格：`NX, NY`（供 GTE 量化）
- 平面與對齊：`EPS`（平面距離容差）、`T_SYNC`（eye/face 合併容差秒）
- BCEA：`BCEA_P`（常用 0.682 或 0.95）
- 視窗：`WIN, STEP`；有效點比例下限：`MIN_VALID_FRAC`
- IVT 凝視：`VT`（UV 速度閾值）、`TMIN`（最短凝視秒）、`USE_FIXRATIO`
- 眨眼：`USE_BLINK`、`BLINK_TH`、`BLINK_MIN`、`BLINK_MAX`、`BLINK_MERGE`
- Hysteresis：`THR_HIGH`、`THR_LOW`、`MIN_ON`、`MIN_OFF`

---

### B. `AOIHeatmap.py` — AOI 佔用熱圖 / 中心五區 / 網格填色

**用法**
```bash
python AOIHeatmap.py [STAMP] [--nx N] [--ny N] [--center-frac F] [--outdir DIR]
```
- `STAMP`：可省略（自動找最新 eye）。
- `--nx / --ny`：AOI 水平／垂直切格數（預設 4×4）。
- `--center-frac`：中心框的寬高比例（UV 0~1，預設 0.6）。
- `--outdir`：輸出資料夾（預設 `.`）。

**輸出**
- `<STAMP>__aoi_heatmap.png`：**NX×NY 矩陣熱圖**（顯示 on‑screen %，含中心框疊線）。
- `<STAMP>__aoi_center5.png`：**中心五區（Center/Top/Bottom/Left/Right）16:9 填色圖**（內嵌百分比）。
- `<STAMP>__aoi_center5.csv`：五區百分比表。
- `<STAMP>__aoi_gridfill_<Nx>x<Ny>.png`：**N×M 16:9 填色與格線**（每格內嵌百分比）。

> 僅統計 UV 在 [0,1]² 且平面距離 < `EPS` 的樣本。


---

### C. `DataRadar.py` — 雷達圖（整體＋Top‑K）

**用法**
```bash
python DataRadar.py <STAMP> [--topk K] [--outdir DIR]
```
- `STAMP`：必填，需有 `<STAMP>__windows.csv`。
- `--topk`：Top‑K 視窗數（預設 5）。
- `--outdir`：輸出資料夾（預設 `.`）。

**輸出**
- `<STAMP>__radar_overall.png`：各指標「**場內百分位**」的**中位數**雷達。  
- `<STAMP>__radar_topK.png`：依 `AttentionScore` 取前 K 段，每段一張雷達。

> 指標軸依序嘗試：`BCEA, GTE, FixRatio, BlinkRate, BlinkFrac`（缺值會自動略過該軸）。  
> 方向一致化：`BCEA/GTE/Blink*` **越小越好 → (1‑p)**，`FixRatio` **越大越好 → p**。


---

## 指標解釋（簡要）

- **BCEA**：雙變量高斯的橢圓面積估計，反映注視點分散程度；質量 `p = BCEA_P`。**越小越集中**。  
- **GTE**：將 UV 量化到 AOI（含 off‑screen）後，移除連續重覆狀態，計算一階馬可夫轉移熵。**越小越穩定**。  
- **FixRatio（IVT）**：以 UV 速度閾值 `VT` 與最短持續 `TMIN` 判定凝視，**視窗中凝視時間比例**。**越大越好**。  
- **BlinkRate / BlinkFrac**：從 `EyesClosed*` 推得眨眼次數與閉眼比例（秒）。多任務情境需審慎解讀。  
- **AttentionScore**：將各指標轉為場內百分位並方向一致後**等權平均**→ 再以百分位**校準到 [0,1]**。  
- **Focused**：對 `AttentionScore` 施以**雙門檻 + 最短持續**，避免抖動。


---

## 常見問題（FAQ）

- **讀不到時間欄位**：請確認檔內有可用時間欄；若欄名不在白名單，請提供數值且單調不減的欄位。  
- **CSV 編碼錯誤**：建議安裝 `charset-normalizer` 與 `chardet`，腳本會自動嘗試多種編碼。  
- **沒有 3D 注視點欄位**：請提供 `Left/RightEye_Pos*` 與 `CombinedGazeForward_*`，由射線與螢幕平面求交。  
- **螢幕內樣本太少**：檢查 `DEFAULT_PLANE`（中心與尺寸）與 `EPS` 是否符合拍攝設定；必要時放寬 `EPS`。  
- **雷達圖沒有輸出**：請先執行 `AttentionGazeKit.py` 產生 `<STAMP>__windows.csv`，再執行 `DataRadar.py`。


---

## 再現性與建議

- **三個腳本務必共用同一組 `DEFAULT_PLANE` 與 `EPS`**，以免 UV 投影不一致。  
- 若要比較注意力與 AOI 熱圖，建議在 `AttentionGazeKit.py` 的 `NX×NY` 與 `AOIHeatmap.py` 的 `--nx/--ny` **使用相同切格**。  
- 每次錄製請將輸出集中到同一個 `results/` 目錄，便於追蹤與比對。


