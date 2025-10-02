#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GazedataAnalysis.py — Run-all orchestrator (split outputs by type)

資料夾結構（建議）：
  專案根目錄/
  ├─ GazedataAnalysis.py
  ├─ Attention Script/
  │  ├─ AttentionGazeKit.py
  │  ├─ AOIHeatmap.py
  │  └─ DataRadar.py
  ├─ Raw Data/
  │  ├─ video_eye_YYYYmmdd_HHMMSS.csv
  │  └─ video_face_YYYYmmdd_HHMMSS.csv
  └─ AnalysisData/
     ├─ pic/<STAMP>/
     └─ CSV/<STAMP>/

流程：
  1) AttentionGazeKit.py → 產生 <STAMP>__windows.csv 與 attention 圖
  2) AOIHeatmap.py      → 產生 Center-5 / GridFill / Heatmap 圖（含 center5.csv）
  3) DataRadar.py       → 產生 Radar（overall + Top-K）

使用方式（於專案根目錄）：
  python GazedataAnalysis.py
  python GazedataAnalysis.py 20251001_211123

常用參數：
  python GazedataAnalysis.py --data-dir "Raw Data" --nx 4 --ny 4 --center-frac 0.6 --topk 5
  python GazedataAnalysis.py 20251001_211123 --extra --no-plot  # 將 --no-plot 傳給 AttentionGazeKit.py

說明：
  - 預設會在 --data-dir（預設 "Raw Data"）內尋找 video_eye_<STAMP>.csv / video_face_<STAMP>.csv。
  - 子腳本執行時的工作目錄（cwd）：
      * AttentionGazeKit.py, AOIHeatmap.py → cwd = --data-dir
      * DataRadar.py → cwd = AnalysisData/_staging/<STAMP>/ 以便讀取 <STAMP>__windows.csv
  - 全部產物先寫入 staging（AnalysisData/_staging/<STAMP>/），完成後再分類移動：
      * 圖片（.png/.jpg/.jpeg/.svg） → AnalysisData/pic/<STAMP>/
      * CSV（.csv）及其他數據檔（例如 .json/.txt） → AnalysisData/CSV/<STAMP>/
"""

import argparse
import subprocess
import sys
from pathlib import Path
import re
import shutil
from typing import List, Optional

# ------ 專案路徑 ------
PROJ_ROOT = Path(__file__).parent.resolve()
SCRIPTS_DIR = PROJ_ROOT / "Attention Script"
ANALYSIS_ROOT = PROJ_ROOT / "AnalysisData"
PIC_ROOT = ANALYSIS_ROOT / "pic"
CSV_ROOT = ANALYSIS_ROOT / "CSV"

# ------ 工具函式 ------
def detect_latest_stamp(data_dir: Path) -> str:
    """從 data_dir 找最新修改時間的 video_eye_*.csv，回傳 STAMP 字串。"""
    cands = sorted(data_dir.glob("video_eye_*.csv"),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not cands:
        raise FileNotFoundError(f"在資料夾不存在任何 'video_eye_*.csv'：{data_dir}")
    m = re.match(r"video_eye_(.+)\.csv$", cands[0].name)
    if not m:
        raise RuntimeError(f"檔名格式不符：{cands[0].name}")
    return m.group(1)

def ensure_scripts() -> None:
    """確認三個分析腳本存在。"""
    for name in ["AttentionGazeKit.py", "AOIHeatmap.py", "DataRadar.py"]:
        p = SCRIPTS_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"找不到腳本：{p}")

def ensure_data_pair(data_dir: Path, stamp: str) -> None:
    """確認需要的 eye/face CSV 是否存在。"""
    eye = data_dir / f"video_eye_{stamp}.csv"
    face = data_dir / f"video_face_{stamp}.csv"
    missing = []
    if not eye.exists():
        missing.append(eye.name)
    if not face.exists():
        missing.append(face.name)
    if missing:
        msg = (
            f"缺少必要檔案於資料夾：{data_dir}\n"
            f"缺少：{', '.join(missing)}\n"
            f"請確認檔名（STAMP）一致、且放在 --data-dir 指定的資料夾內。"
        )
        raise FileNotFoundError(msg)

def run_script(script: Path, args: List[str], cwd: Optional[Path] = None) -> None:
    """以指定 cwd 執行子腳本，非 0 結束碼則中止。"""
    cmd = [sys.executable, str(script), *args]
    print(">>>", " ".join([repr(x) for x in cmd]))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"[ERROR] {script.name} exited with code {proc.returncode}")

def move_outputs(staging: Path, stamp: str) -> None:
    """將 staging 內的輸出分類搬到 AnalysisData/pic/<STAMP>/ 與 AnalysisData/CSV/<STAMP>/。"""
    pic_dst = PIC_ROOT / stamp
    csv_dst = CSV_ROOT / stamp
    pic_dst.mkdir(parents=True, exist_ok=True)
    csv_dst.mkdir(parents=True, exist_ok=True)

    img_exts = {".png", ".jpg", ".jpeg", ".svg"}
    for item in staging.iterdir():
        if item.is_dir():
            continue
        ext = item.suffix.lower()
        if ext in img_exts:
            shutil.move(str(item), str(pic_dst / item.name))
        elif ext == ".csv":
            shutil.move(str(item), str(csv_dst / item.name))
        else:
            # 其他數據檔先一律放 CSV 夾，之後若有需要可再細分
            shutil.move(str(item), str(csv_dst / item.name))

    # 試著清 staging 空夾
    try:
        staging.rmdir()
    except OSError:
        pass

# ------ 主流程 ------
def main():
    parser = argparse.ArgumentParser(
        description="Run gaze analyses; split outputs into AnalysisData/{pic,CSV}/<STAMP>/"
    )
    parser.add_argument(
        "stamp",
        nargs="?",
        help="錄製 STAMP（匹配 video_eye_<STAMP>.csv）。若省略，會在 --data-dir 內自動取最新。"
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJ_ROOT / "Raw Data"),
        help="原始 CSV 資料夾，預設 'Raw Data'"
    )
    parser.add_argument("--nx", type=int, default=4, help="AOIHeatmap.py 的 AOI 欄數（預設 4）")
    parser.add_argument("--ny", type=int, default=4, help="AOIHeatmap.py 的 AOI 列數（預設 4）")
    parser.add_argument("--center-frac", type=float, default=0.6, help="Center-5 框的 UV 尺寸（預設 0.6）")
    parser.add_argument("--topk", type=int, default=5, help="DataRadar.py 的 Top-K（預設 5）")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="多餘參數傳給 AttentionGazeKit.py（例如：--extra --no-plot）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到資料夾：{data_dir}")

    # 確保三個腳本存在
    ensure_scripts()

    # 解析 STAMP（若未指定則自動偵測 Raw Data/ 最新）
    stamp = args.stamp or detect_latest_stamp(data_dir)

    # 檢查 eye/face 配對檔是否存在（你的 AttentionGazeKit 版本需要兩個檔）
    ensure_data_pair(data_dir, stamp)

    # 建 staging 夾（先讓所有輸出落地於 staging，再分類搬移）
    staging = ANALYSIS_ROOT / "_staging" / stamp
    staging.mkdir(parents=True, exist_ok=True)

    # 子腳本路徑
    gaze_kit = SCRIPTS_DIR / "AttentionGazeKit.py"
    aoi_heat = SCRIPTS_DIR / "AOIHeatmap.py"
    radar = SCRIPTS_DIR / "DataRadar.py"

    # Step 1：AttentionGazeKit（在 data_dir 當前目錄下執行，以便找到 CSV）
    extra = args.extra or []
    run_script(gaze_kit, [stamp, "--outdir", str(staging), *extra], cwd=data_dir)

    # Step 2：AOIHeatmap（同樣以 data_dir 為 cwd，使其能讀 CSV）
    run_script(
        aoi_heat,
        [
            stamp,
            "--nx", str(args.nx),
            "--ny", str(args.ny),
            "--center-frac", str(args.center_frac),
            "--outdir", str(staging),
        ],
        cwd=data_dir
    )

    # Step 3：DataRadar（需要讀 <STAMP>__windows.csv，因此以 staging 為 cwd）
    run_script(
        radar,
        [
            stamp,
            "--topk", str(args.topk),
            "--outdir", str(staging),
        ],
        cwd=staging
    )

    # 分流搬移
    move_outputs(staging, stamp)

    print("\nAll done ✅")
    print(f"Images: {(PIC_ROOT / stamp).resolve()}")
    print(f"CSVs:   {(CSV_ROOT / stamp).resolve()}")

if __name__ == "__main__":
    # 確保輸出基礎資料夾存在
    for d in [SCRIPTS_DIR, ANALYSIS_ROOT, PIC_ROOT, CSV_ROOT]:
        d.mkdir(parents=True, exist_ok=True)
    main()
