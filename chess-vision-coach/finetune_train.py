"""
finetune_train.py — CPU-optimized fine-tuning runner.

Fine-tunes a YOLO detector on the prepared chess dataset (finetune_data/).
Defaults are tuned for a CPU-only box: nano model, modest image size,
RAM caching, early stopping. Override via env vars if you have a GPU.

  CVC_FT_MODEL   base weights        (default yolov8n.pt)
  CVC_FT_IMGSZ   training image size (default 416)
  CVC_FT_EPOCHS  max epochs          (default 20)
  CVC_FT_BATCH   batch size          (default 16)
  CVC_FT_EXPORT  exported .onnx name (default yolov8n-chess-finetuned.onnx)
  CVC_FT_RUN     run directory name  (default yolo_chess_cpu)

Outputs:
  runs/detect/chess_finetune/<name>/weights/best.pt   best checkpoint
  chess_vision/models/yolov8n-chess-finetuned.onnx     exported ONNX (drop-in)
"""

import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

MODEL  = os.environ.get("CVC_FT_MODEL", "yolov8n.pt")
IMGSZ  = int(os.environ.get("CVC_FT_IMGSZ", "416"))
EPOCHS = int(os.environ.get("CVC_FT_EPOCHS", "20"))
BATCH  = int(os.environ.get("CVC_FT_BATCH", "16"))
DATA   = HERE / "finetune_data" / "dataset.yaml"
RUN    = os.environ.get("CVC_FT_RUN", "yolo_chess_cpu")

EXPORT_NAME = os.environ.get("CVC_FT_EXPORT", "yolov8n-chess-finetuned.onnx")


def main() -> None:
    if not DATA.exists():
        print(f"ERROR: {DATA} not found. Run prepare_finetune_dataset.py first.")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"== fine-tune ==  model={MODEL} imgsz={IMGSZ} epochs={EPOCHS} batch={BATCH}")
    t0 = time.time()

    model = YOLO(MODEL)
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="cpu",
        workers=4,
        cache="ram",        # 388 small imgs fit easily in 15GB
        patience=5,         # early-stop once val mAP plateaus
        lr0=0.002,          # fine-tune LR (lower than the 0.01 default)
        lrf=0.01,
        warmup_epochs=2,
        # chess-appropriate augmentation
        degrees=8,
        fliplr=0.5,
        flipud=0.0,         # board orientation matters; never flip vertically
        mosaic=0.0,         # mosaic hurts grid-structured boards
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
        plots=True,
        project="chess_finetune",
        name=RUN,
        exist_ok=True,
        verbose=True,
    )

    mins = (time.time() - t0) / 60.0
    print(f"\n== training done in {mins:.1f} min ==")

    # Ultralytics resolves the save dir relative to its runs_dir (typically
    # runs/detect/<project>/<name>), so derive best.pt from the trainer
    # rather than guessing the path.
    best = Path(model.trainer.best)
    if not best.exists():
        print(f"ERROR: {best} not found")
        sys.exit(1)

    print("== exporting to ONNX ==")
    YOLO(str(best)).export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True)

    onnx_src = best.with_suffix(".onnx")
    onnx_dst = HERE / "chess_vision" / "models" / EXPORT_NAME
    shutil.copy(onnx_src, onnx_dst)
    print(f"== exported -> {onnx_dst} ==")


if __name__ == "__main__":
    main()
