"""Слой 3: Детектор военных объектов на видео через YOLO-World (ONNX).

MilitaryDetector:
  - при старте определяет железо и выбирает модель;
  - нормализует имя модели под реально существующие файлы в models/;
  - загружает YOLO (ultralytics сам подхватит ONNX Runtime с CUDA/CPU);
  - анализирует видео с заданным шагом кадров и сохраняет JSON-отчёт.
"""
import os
import json
import datetime
from pathlib import Path

import cv2

from core import hardware, backend
from core.classes import MILITARY_CLASSES


def _normalize_model_name(name: str) -> str:
    """Приводит имя модели из backend.choose_model к реально существующим файлам.

    Правила из ТЗ:
      - 'world'  -> 'worldv2'   (у нас только v2-веса)
      - размер 'n' -> 's'        (nano-модели нет, понижаем до small)
      - размер 'x' -> 'l'       (xlarge нет, понижаем до large)
    """
    # world -> worldv2
    if "world" in name and "worldv2" not in name:
        name = name.replace("world", "worldv2")

    # размер: вытащим букву размера между yolov8 и -worldv2
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)  # yolov8n-worldv2, .onnx

    if stem.startswith("yolov8") and "-worldv2" in stem:
        size_char = stem[len("yolov8")]            # 'n', 's', 'm', 'l', 'x'
        size_map = {"n": "s", "x": "l"}
        new_size = size_map.get(size_char, size_char)
        stem = f"yolov8{new_size}-worldv2"

    return f"{stem}{ext}"


class MilitaryDetector:
    """Детектор военных объектов на видео."""

    def __init__(self):
        # Определяем железо и выбираем модель/провайдеры
        self.hw = hardware.detect_all()
        self.gpu = self.hw.get("gpu")
        self.cpu = self.hw.get("cpu", {})

        raw_name = backend.choose_model(self.gpu)
        self.model_name = _normalize_model_name(raw_name)
        self.providers = backend.choose_providers(self.gpu)

        # Путь к файлу модели
        self.model_path = str(Path("models") / self.model_name)

        # Загружаем модель (ultralytics сам подхватит ONNX Runtime)
        # Импорт внутри метода, чтобы не тянуть тяжёлый ultralytics при импорте модуля
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

        # Шаги кадров из .env (с дефолтами)
        self.frame_step = int(os.getenv("FRAME_STEP", "30"))
        self.drone_frame_step = int(os.getenv("DRONE_FRAME_STEP", "15"))

    def analyze_video(self, path, drone_mode=False, confidence=0.25,
                      progress_callback=None):
        """Анализирует видео и возвращает список моментов с находками.

        Аргументы:
          path             — путь к видеофайлу;
          drone_mode       — True => шаг кадров DRONE_FRAME_STEP (15);
          confidence       — порог уверенности детекции;
          progress_callback(percent, timestamp_str, objects) — опциональный колбэк для GUI.

        Возвращает:
          list[dict] — моменты вида
            {timestamp, objects:[{class, confidence, bbox}]}
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = self.drone_frame_step if drone_mode else self.frame_step

        moments = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                # Детекция
                results = self.model.predict(
                    frame,
                    conf=confidence,
                    verbose=False,
                    classes=None,  # используем все классы, вшитые в ONNX
                )

                objects = []
                if results:
                    r = results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
                            cls_name = (MILITARY_CLASSES[cls_id]
                                        if 0 <= cls_id < len(MILITARY_CLASSES)
                                        else f"class_{cls_id}")
                            objects.append({
                                "class": cls_name,
                                "confidence": round(conf, 3),
                                "bbox": [round(v, 1) for v in xyxy],
                            })

                if objects:
                    timestamp_sec = frame_idx / fps
                    ts = str(datetime.timedelta(seconds=int(timestamp_sec)))
                    moments.append({
                        "timestamp": ts,
                        "frame": frame_idx,
                        "objects": objects,
                    })

                # Прогресс для GUI
                if progress_callback is not None:
                    percent = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    ts_str = str(datetime.timedelta(seconds=int(frame_idx / fps)))
                    try:
                        progress_callback(min(percent, 100.0), ts_str, objects)
                    except Exception:
                        pass

            frame_idx += 1

        cap.release()

        # Сохраняем JSON-отчёт в output/
        self._save_report(path, moments)
        return moments

    def _save_report(self, video_path, moments):
        """Сохраняет JSON-отчёт в папку output/."""
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)

        video_stem = Path(video_path).stem
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"{video_stem}_report_{ts}.json"

        report = {
            "video": str(video_path),
            "model": self.model_name,
            "hardware": self.hw,
            "created_at": datetime.datetime.now().isoformat(),
            "moments_count": len(moments),
            "moments": moments,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.last_report_path = str(out_file)