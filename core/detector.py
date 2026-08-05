"""Слой 3: Детектор военных объектов на видео через YOLO-World (ONNX).

MilitaryDetector:
  - при старте определяет железо и выбирает модель;
  - нормализует имя модели под реально существующие файлы в models/;
  - загружает YOLO (ultralytics сам подхватит ONNX Runtime с CUDA/CPU);
  - анализирует видео с заданным шагом кадров и сохраняет JSON-отчёт.
"""
import os
import json
import time
import datetime
from pathlib import Path

import cv2

from core import hardware, backend
from core.classes import MILITARY_CLASSES, ru


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
        # task="detect" убирает warning "Unable to automatically guess model task"
        self.model = YOLO(self.model_path, task="detect")

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
          progress_callback(percent, timestamp_str, objects, extra_log=None,
                            eta_sec=None) — опциональный колбэк для GUI.

        Возвращает:
          list[dict] — моменты вида
            {timestamp, objects:[{class, confidence, bbox}]}
          class — на русском (через ru()).
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = self.drone_frame_step if drone_mode else self.frame_step

        moments = []
        frame_idx = 0

        # ── ETA: измеряем время обработки последних 10 кадров ──
        _frame_times = []  # время обработки каждого анализируемого кадра
        _last_time = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                # Замер времени начала обработки кадра (для ETA)
                _t_start = time.time()

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
                            # Переводим класс на русский для отображения
                            cls_name_ru = ru(cls_name)
                            objects.append({
                                "class": cls_name_ru,
                                "class_en": cls_name,
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

                # Замер времени конца обработки кадра
                _t_end = time.time()
                _frame_times.append(_t_end - _t_start)
                # Оставляем только последние 10 замеров
                if len(_frame_times) > 10:
                    _frame_times.pop(0)

                # Прогресс для GUI + ETA (каждый 50-й кадр)
                if progress_callback is not None:
                    percent = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    ts_str = str(datetime.timedelta(seconds=int(frame_idx / fps)))

                    # ETA: среднее время кадра × оставшихся кадров
                    eta_sec = None
                    if frame_idx % 50 == 0 and _frame_times and total_frames > 0:
                        avg_per_frame = sum(_frame_times) / len(_frame_times)
                        # Учитываем шаг: анализируем каждый step-й кадр
                        remaining_analyze = max(0, (total_frames - frame_idx) // step)
                        eta_sec = int(avg_per_frame * remaining_analyze)

                    try:
                        progress_callback(min(percent, 100.0), ts_str, objects,
                                          eta_sec=eta_sec)
                    except TypeError:
                        # Совместимость со старым колбэком без eta_sec
                        try:
                            progress_callback(min(percent, 100.0), ts_str, objects)
                        except Exception:
                            pass
                    except Exception:
                        pass

            frame_idx += 1

        cap.release()

        # Экспортируем стоп-кадры в папку output/<video_stem>_frames/
        frames_dir = self._export_frames(path, moments, progress_callback)

        # Сохраняем JSON-отчёт в output/
        self._save_report(path, moments, frames_dir=frames_dir)
        return moments

    def _export_frames(self, video_path, moments, progress_callback=None):
        """Экспортирует стоп-кадры моментов в папку output/<video_stem>_frames/.

        Для каждого момента:
          - достаёт кадр по индексу;
          - рисует все bbox с именами классов;
          - сохраняет JPEG с именем вида "00-00-13_military_boat_0.35.jpg".

        Возвращает путь к созданной папке (str) или None, если моментов нет.
        """
        if not moments:
            return None

        # Папка: output/<имя_видео>_frames/
        video_stem = Path(video_path).stem
        frames_dir = Path("output") / f"{video_stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        for moment in moments:
            frame_idx = moment["frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Рисуем все bbox с именами классов
            annotated = self._draw_boxes(frame, moment["objects"])

            # Формируем имя файла
            filename = self._build_frame_filename(moment)
            out_path = frames_dir / filename
            cv2.imwrite(str(out_path), annotated,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        cap.release()

        if progress_callback is not None:
            try:
                progress_callback(
                    100.0, "", [],
                    extra_log=f"🖼 Стоп-кадры сохранены: {frames_dir}",
                )
            except Exception:
                # Совместимость со старым колбэком без extra_log
                pass

        return str(frames_dir)

    def _draw_boxes(self, frame, objects):
        """Рисует рамки bbox и имена классов на кадре (для экспорта)."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        thickness = max(1, int(min(h, w) / 300))

        # Цвета для разных классов (BGR)
        colors = [
            (0x44, 0x44, 0xff), (0x44, 0xff, 0x44), (0xff, 0x44, 0x44),
            (0x44, 0xff, 0xff), (0xff, 0x44, 0xff), (0xff, 0xff, 0x44),
        ]

        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            b, g, r = colors[i % len(colors)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (b, g, r), thickness)

            label = f"{obj['class']} {obj['confidence']:.2f}"
            font_scale = max(0.4, min(h, w) / 800)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4),
                          (x1 + tw + 4, y1), (b, g, r), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), 1, cv2.LINE_AA)

        return annotated

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Удаляет недопустимые символы в именах файлов Windows."""
        for ch in ':*?"<>|':
            name = name.replace(ch, "")
        return name.strip().rstrip(".")

    @staticmethod
    def _format_timestamp(ts: str) -> str:
        """Преобразует '0:00:13' -> '00-00-13' (ЧЧ-ММ-СС, без двоеточий)."""
        parts = ts.split(":")
        # Дополняем каждую часть до 2 цифр
        parts = [p.zfill(2) for p in parts]
        return "-".join(parts)

    def _build_frame_filename(self, moment) -> str:
        """Формирует имя JPEG-файла для момента.

        Пример: "00-00-13_military_boat_0.35.jpg"
        Если объектов несколько: "00-00-13_military_boat_и_ещё_2.jpg"
        """
        ts = self._format_timestamp(moment["timestamp"])
        objects = moment["objects"]
        first = objects[0]
        first_class = first["class"].replace(" ", "_")
        first_conf = first["confidence"]

        if len(objects) == 1:
            base = f"{ts}_{first_class}_{first_conf:.2f}"
        else:
            extra = len(objects) - 1
            base = f"{ts}_{first_class}_и_ещё_{extra}"

        filename = f"{self._sanitize_filename(base)}.jpg"
        return filename

    def _save_report(self, video_path, moments, frames_dir=None):
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
        if frames_dir:
            report["frames_dir"] = frames_dir

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.last_report_path = str(out_file)
