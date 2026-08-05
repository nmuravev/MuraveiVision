"""PRO-версия MuraveiVision (PySide6 + QWebEngineView).

Скелет: QSplitter 3 зоны [300, 900, 400]:
  - СЛЕВА: проводник с фильтром видео;
  - ЦЕНТР: плеер (cv2→QImage→QLabel, QThread, play/pause, seek, in/out маркеры);
  - СПРАВА: QWebEngineView с HTML v2.

Взаимодействие:
  - После анализа → HTML v2 грузится в QWebEngineView справа;
  - Клик по карточке в HTML → плеер прыгает на таймкод (через QWebChannel);
  - "💾 Сохранить выбранные" → JS→Python через QWebChannel → папка output/<video>_selected_<ts>/.
"""
import os
import sys
import json
import time
import threading
import datetime
from pathlib import Path

import cv2

from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QObject, Slot, QUrl, QSize,
)
from PySide6.QtGui import QImage, QPixmap, QAction, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QLineEdit, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QMessageBox, QProgressBar, QGroupBox,
    QFrame,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel

from core import hardware, backend
from core.detector import MilitaryDetector
from core.classes import MILITARY_CLASSES, ru
from core import nvidia_client


# ── AMOLED-стиль (QSS) ──
_QSS = """
QMainWindow, QWidget { background: #000000; color: #E0E0E0; }
QLabel { color: #E0E0E0; }
QPushButton {
    background: #00E5FF; color: #000000; border: none;
    padding: 8px 16px; border-radius: 6px; font-weight: bold;
}
QPushButton:hover { background: #00B8D4; }
QPushButton:disabled { background: #333; color: #666; }
QPushButton#success { background: #00FF88; }
QPushButton#success:hover { background: #00CC6E; }
QLineEdit {
    background: #0D0D0D; border: 1px solid #1E1E1E; color: #E0E0E0;
    padding: 6px; border-radius: 4px;
}
QSlider::groove:horizontal {
    background: #1E1E1E; height: 6px; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #00E5FF; width: 16px; margin: -5px 0; border-radius: 8px;
}
QProgressBar {
    background: #1E1E1E; border: none; border-radius: 3px; text-align: center;
}
QProgressBar::chunk { background: #00E5FF; border-radius: 3px; }
QTreeWidget {
    background: #0D0D0D; border: 1px solid #1E1E1E; color: #E0E0E0;
}
QTreeWidget::item:selected { background: #00E5FF; color: #000000; }
QCheckBox { color: #E0E0E0; }
QGroupBox {
    border: 1px solid #1E1E1E; border-radius: 8px; margin-top: 12px;
    padding-top: 12px; color: #00E5FF; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
"""


# ── Мост Python ↔ JS (QWebChannel) ──
class PyBridge(QObject):
    """Мост для вызовов из JavaScript HTML-отчёта.

    JS вызывает window.pyBridge.seekToMoment(frame, ts) → плеер прыгает.
    JS вызывает window.pyBridge.saveSelected(base64_json) → Python сохраняет папку.
    """
    seekRequested = Signal(int, str)  # frame_idx, timestamp
    saveRequested = Signal(str)       # JSON-строка с выбранными моментами

    @Slot(int, str)
    def seekToMoment(self, frame_idx, timestamp):
        """Вызывается из JS при клике по карточке — плеер прыгает на таймкод."""
        self.seekRequested.emit(frame_idx, timestamp)

    @Slot(str)
    def saveSelected(self, selected_json):
        """Вызывается из JS при '💾 Сохранить выбранные' — Python создаёт папку."""
        self.saveRequested.emit(selected_json)


# ── Поток воспроизведения видео ──
class VideoPlayerThread(QThread):
    """Поток воспроизведения видео: читает кадры cv2 → QImage → сигнал."""
    frameReady = Signal(QImage)
    positionChanged = Signal(int, float)  # frame_idx, seconds

    def __init__(self, video_path=""):
        super().__init__()
        self.video_path = video_path
        self._running = False
        self._paused = False
        self._seek_frame = -1
        self.fps = 25.0
        self.total_frames = 0

    def set_video(self, path):
        """Устанавливает путь к видео."""
        self.video_path = path
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def play(self):
        """Запускает/возобновляет воспроизведение."""
        self._paused = False
        if not self._running:
            self._running = True
            self.start()

    def pause(self):
        """Приостанавливает воспроизведение."""
        self._paused = True

    def seek(self, frame_idx):
        """Перематывает к указанному кадру."""
        self._seek_frame = frame_idx

    def stop(self):
        """Останавливает поток."""
        self._running = False
        self.wait(2000)

    def run(self):
        """Главный цикл потока: читает кадры и emits."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        frame_idx = 0
        delay_ms = int(1000 / self.fps)

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            # Обработка seek
            if self._seek_frame >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_frame)
                frame_idx = self._seek_frame
                self._seek_frame = -1

            ret, frame = cap.read()
            if not ret:
                # Конец видео — зацикливаем или останавливаем
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            # BGR → RGB → QImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            self.frameReady.emit(qimg)
            self.positionChanged.emit(frame_idx, frame_idx / self.fps)

            frame_idx += 1
            # Задержка для реалтайм-воспроизведения
            time.sleep(delay_ms / 1000.0)

        cap.release()


# ── Главное окно PRO ──
class ProMainWindow(QMainWindow):
    """Главное окно PRO-версии: QSplitter 3 зоны."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 MuraveiVision PRO — Military Object Detector")
        self.resize(1600, 900)
        self.setStyleSheet(_QSS)

        # Детектор (один экземпляр)
        self.detector = MilitaryDetector()

        # Состояние
        self.video_path = ""
        self.last_html_path = ""
        self.last_moments = []
        self.in_marker = 0      # in-маркер (кадр)
        self.out_marker = 0     # out-маркер (кадр)
        self.export_frames = False  # экспорт кадров (по чекбоксу)

        # Мост Python ↔ JS
        self.bridge = PyBridge()
        self.bridge.seekRequested.connect(self._on_seek_from_html)
        self.bridge.saveRequested.connect(self._on_save_selected)

        # Поток плеера
        self.player_thread = VideoPlayerThread()
        self.player_thread.frameReady.connect(self._on_frame)
        self.player_thread.positionChanged.connect(self._on_position)

        # ── Построение UI ──
        self._build_ui()

        # Лог
        self._log("✅ PRO-версия готова")
        self._log(f"🤖 Модель: {self.detector.model_name}")
        self._log(f"🖥️ Провайдеры: {', '.join(self.detector.providers)}")

        # Прокси-политика
        proxy = os.getenv("NVIDIA_PROXY", "")
        if not proxy:
            self._log("☁️ Облако отключено: политика — только через прокси")

    def _build_ui(self):
        """Строит интерфейс: QSplitter 3 зоны."""
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ── СЛЕВА: проводник с фильтром видео ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        left_layout.addWidget(QLabel("📁 Проводник"))
        self.filter_edit = QLineEdit(placeholderText="🔍 Фильтр видео...")
        self.filter_edit.textChanged.connect(self._on_filter)
        left_layout.addWidget(self.filter_edit)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self._on_tree_double_click)
        left_layout.addWidget(self.tree, stretch=1)

        # Кнопка выбора папки
        browse_btn = QPushButton("📂 Выбрать папку")
        browse_btn.clicked.connect(self._on_browse_folder)
        left_layout.addWidget(browse_btn)

        splitter.addWidget(left)

        # ── ЦЕНТР: плеер ──
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)

        # Видео-дисплей
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setText("🎬 Видео не загружено")
        self.video_label.setStyleSheet("background:#0D0D0D; border:1px solid #1E1E1E; border-radius:8px;")
        self.video_label.setMinimumSize(640, 360)
        center_layout.addWidget(self.video_label, stretch=1)

        # Контролы плеера
        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶️")
        self.play_btn.setFixedWidth(50)
        self.play_btn.clicked.connect(self._on_play_pause)
        controls.addWidget(self.play_btn)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.sliderMoved.connect(self._on_seek_slider)
        controls.addWidget(self.seek_slider, stretch=1)

        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setStyleSheet("font-family: Consolas, monospace; color: #00E5FF;")
        controls.addWidget(self.time_label)
        center_layout.addLayout(controls)

        # In/out маркеры
        markers = QHBoxLayout()
        in_btn = QPushButton("[I] In")
        in_btn.clicked.connect(self._on_set_in)
        markers.addWidget(in_btn)
        out_btn = QPushButton("[O] Out")
        out_btn.clicked.connect(self._on_set_out)
        markers.addWidget(out_btn)
        self.markers_label = QLabel("In: 0 | Out: 0")
        self.markers_label.setStyleSheet("font-family: Consolas, monospace; color: #8A8A8A;")
        markers.addWidget(self.markers_label)
        center_layout.addLayout(markers)

        # Настройки анализа
        settings_group = QGroupBox("Настройки анализа")
        settings_layout = QHBoxLayout(settings_group)

        self.drone_check = QCheckBox("🎯 🚁 Режим дрона")
        settings_layout.addWidget(self.drone_check)

        self.cloud_check = QCheckBox("☁️ Облако")
        self.cloud_check.toggled.connect(self._on_cloud_toggle)
        settings_layout.addWidget(self.cloud_check)

        # Маскированное поле ключа
        self.key_edit = QLineEdit(placeholderText="nvapi-...")
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setText(os.getenv("NVIDIA_API_KEY", ""))
        settings_layout.addWidget(self.key_edit, stretch=1)

        # Иконка 👁 — показать/скрыть ключ
        self.key_toggle_btn = QPushButton("👁")
        self.key_toggle_btn.setFixedWidth(40)
        self.key_toggle_btn.clicked.connect(self._toggle_key_visibility)
        settings_layout.addWidget(self.key_toggle_btn)

        self.export_check = QCheckBox("💾 Экспорт кадров")
        settings_layout.addWidget(self.export_check)

        center_layout.addWidget(settings_group)

        # Кнопка анализа + прогресс
        action_row = QHBoxLayout()
        self.analyze_btn = QPushButton("▶️ Анализ")
        self.analyze_btn.setObjectName("success")
        self.analyze_btn.clicked.connect(self._on_analyze)
        action_row.addWidget(self.analyze_btn)

        self.batch_btn = QPushButton("📁 Пакет")
        self.batch_btn.clicked.connect(self._on_batch)
        action_row.addWidget(self.batch_btn)

        center_layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        center_layout.addWidget(self.progress)

        self.progress_label = QLabel("0%")
        self.progress_label.setStyleSheet("font-family: Consolas, monospace; color: #8A8A8A;")
        center_layout.addWidget(self.progress_label)

        # Лог
        self.log_edit = QLineEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("background:#0D0D0D; border:1px solid #1E1E1E; font-family: Consolas, monospace;")
        center_layout.addWidget(self.log_edit)

        splitter.addWidget(center)

        # ── СПРАВА: QWebEngineView с HTML v2 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("📄 HTML-отчёт v2"))
        self.web_view = QWebEngineView()
        right_layout.addWidget(self.web_view, stretch=1)

        # Кнопка "Открыть HTML" (внешний браузер)
        open_html_btn = QPushButton("🌐 Открыть в браузере")
        open_html_btn.clicked.connect(self._on_open_html_external)
        right_layout.addWidget(open_html_btn)

        splitter.addWidget(right)

        # Размеры зон [300, 900, 400]
        splitter.setSizes([300, 900, 400])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

    # ── Лог ──
    def _log(self, msg):
        """Добавляет сообщение в лог."""
        self.log_edit.setText(msg)

    # ── Проводник ──
    def _on_browse_folder(self):
        """Выбор папки для проводника."""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с видео")
        if folder:
            self._populate_tree(folder)

    def _populate_tree(self, folder):
        """Заполняет дерево проводника видеофайлами."""
        exts = (".mp4", ".avi", ".mov", ".mkv")
        self.tree.clear()
        try:
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(exts):
                    item = QTreeWidgetItem([f])
                    item.setData(0, Qt.UserRole, os.path.join(folder, f))
                    self.tree.addTopLevelItem(item)
        except Exception as e:
            self._log(f"❌ Ошибка чтения папки: {e}")

    def _on_filter(self, text):
        """Фильтрует дерево по тексту."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setHidden(text.lower() not in item.text(0).lower())

    def _on_tree_double_click(self, item, column):
        """Двойной клик по видео — загружаем в плеер."""
        path = item.data(0, Qt.UserRole)
        if path and os.path.exists(path):
            self._load_video(path)

    def _load_video(self, path):
        """Загружает видео в плеер."""
        self.video_path = path
        self.player_thread.set_video(path)
        self.seek_slider.setRange(0, self.player_thread.total_frames)
        self.player_thread.play()
        self.play_btn.setText("⏸️")
        self._log(f"🎬 Загружено: {os.path.basename(path)}")

    # ── Плеер ──
    def _on_frame(self, qimg):
        """Обновляет дисплей плеера новым кадром."""
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def _on_position(self, frame_idx, seconds):
        """Обновляет позицию слайдера и таймкода."""
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_idx)
        self.seek_slider.blockSignals(False)

        # Форматирование времени ЧЧ:ММ:СС
        ts = str(datetime.timedelta(seconds=int(seconds)))
        total_ts = str(datetime.timedelta(
            seconds=int(self.player_thread.total_frames / self.player_thread.fps)))
        self.time_label.setText(f"{ts} / {total_ts}")

    def _on_play_pause(self):
        """Play/Pause."""
        if self.player_thread._paused:
            self.player_thread.play()
            self.play_btn.setText("⏸️")
        else:
            self.player_thread.pause()
            self.play_btn.setText("▶️")

    def _on_seek_slider(self, frame_idx):
        """Перематывает к кадру по слайдеру."""
        self.player_thread.seek(frame_idx)

    def _on_set_in(self):
        """Устанавливает in-маркер."""
        self.in_marker = self.seek_slider.value()
        self.markers_label.setText(
            f"In: {self.in_marker} | Out: {self.out_marker}")

    def _on_set_out(self):
        """Устанавливает out-маркер."""
        self.out_marker = self.seek_slider.value()
        self.markers_label.setText(
            f"In: {self.in_marker} | Out: {self.out_marker}")

    # ── Маскировка ключа ──
    def _toggle_key_visibility(self):
        """Переключает видимость API-ключа."""
        if self.key_edit.echoMode() == QLineEdit.Password:
            self.key_edit.setEchoMode(QLineEdit.Normal)
            self.key_toggle_btn.setText("🙈")
        else:
            self.key_edit.setEchoMode(QLineEdit.Password)
            self.key_toggle_btn.setText("👁")

    def _on_cloud_toggle(self, checked):
        """Проверяет прокси-политику при включении облака."""
        if checked:
            proxy = os.getenv("NVIDIA_PROXY", "")
            if not proxy:
                self._log("☁️ Облако отключено: политика — только через прокси")
                self.cloud_check.setChecked(False)

    # ── Анализ ──
    def _on_analyze(self):
        """Запуск одиночного анализа в отдельном потоке."""
        if not self.video_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите видео!")
            return
        self.analyze_btn.setEnabled(False)
        self.export_frames = self.export_check.isChecked()

        # Сохраняем ключ в .env, если введён
        key = self.key_edit.text().strip()
        if key:
            os.environ["NVIDIA_API_KEY"] = key

        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        """Анализ видео в отдельном потоке."""
        try:
            self.progress.setValue(0)
            self._log(f"▶️ Запуск анализа: {os.path.basename(self.video_path)}")

            results = self.detector.analyze_video(
                self.video_path,
                drone_mode=self.drone_check.isChecked(),
                confidence=0.35,
                progress_callback=self._on_progress,
            )

            self.last_moments = results
            self._log(f"✅ Найдено моментов: {len(results)}")

            # Облачная проверка (с прокси-политикой)
            cloud_annotations = {}
            if self.cloud_check.isChecked() and results:
                proxy = os.getenv("NVIDIA_PROXY", "")
                if proxy and nvidia_client.is_available():
                    cloud_annotations = self._run_cloud_analysis(results)
                else:
                    self._log("☁️ Облако отключено: политика — только через прокси")

            # Генерация HTML v2
            from core.report_html import generate_html
            settings = {
                "drone_mode": self.drone_check.isChecked(),
                "confidence": 0.35,
                "frame_step": self.detector.frame_step,
                "drone_frame_step": self.detector.drone_frame_step,
            }
            report_data = {
                "model": self.detector.model_name,
                "hardware": self.detector.hw,
                "created_at": datetime.datetime.now().isoformat(),
                "moments_count": len(results),
            }
            html_path = generate_html(
                self.video_path, results,
                report_data=report_data,
                settings=settings,
                cloud_annotations=cloud_annotations,
            )
            self.last_html_path = html_path
            self._log(f"📄 HTML-отчёт: {html_path}")

            # Загружаем HTML в QWebEngineView (с QWebChannel)
            QTimer.singleShot(0, lambda: self._load_html(html_path))

        except Exception as e:
            self._log(f"❌ Ошибка: {e}")
        finally:
            QTimer.singleShot(0, lambda: self.analyze_btn.setEnabled(True))

    def _on_progress(self, percent, timestamp_str, objects, extra_log=None, eta_sec=None):
        """Колбэк прогресса из детектора."""
        def _update():
            self.progress.setValue(int(percent))
            eta_str = ""
            if eta_sec is not None and eta_sec >= 0:
                mins = eta_sec // 60
                secs = eta_sec % 60
                eta_str = f" • осталось {mins}м {secs}с"
            self.progress_label.setText(f"{int(percent)}%{eta_str}")
            if extra_log:
                self._log(extra_log)
            elif objects:
                classes = [o["class"] for o in objects]
                self._log(f"⏱️ [{int(percent)}%]{eta_str} | {timestamp_str} | "
                          f"{len(objects)} объектов: {', '.join(classes[:5])}")
        QTimer.singleShot(0, _update)

    def _run_cloud_analysis(self, results):
        """Облачная проверка до 5 кропов с наименьшей confidence."""
        if not nvidia_client.is_available():
            return {}

        all_objs = []
        for moment in results:
            for obj in moment["objects"]:
                all_objs.append((moment["frame"], obj, moment))

        all_objs.sort(key=lambda x: x[1]["confidence"])
        to_check = all_objs[:5]
        if not to_check:
            return {}

        self._log(f"☁️ Облачная проверка {len(to_check)} кропов...")
        annotations = {}
        cap = cv2.VideoCapture(self.video_path)

        for frame_idx, obj, moment in to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            self._log(f"  ☁️ Проверка: {obj['class']} (conf={obj['confidence']:.2f})...")
            answer = nvidia_client.analyze_crop(frame, obj["bbox"])
            if answer:
                annotations[frame_idx] = answer
                self._log(f"  ✅ {obj['class']}: {answer[:100]}...")
            else:
                self._log(f"  ⚠️ {obj['class']}: нет ответа от слоя 4")

        cap.release()
        return annotations

    # ── Загрузка HTML в QWebEngineView ──
    def _load_html(self, html_path):
        """Загружает HTML-отчёт в QWebEngineView с QWebChannel."""
        # Создаём канал и регистрируем мост
        self.channel = QWebChannel()
        self.channel.registerObject("pyBridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        # Загружаем файл
        url = QUrl.fromLocalFile(html_path)
        self.web_view.load(url)

        # Внедряем JS-хук для QWebChannel (после загрузки)
        self.web_view.loadFinished.connect(self._on_html_loaded)

    def _on_html_loaded(self, ok):
        """Вызывается после загрузки HTML — внедряем JS-хук для QWebChannel."""
        if not ok:
            return
        # JS: переопределяем window.pyBridge для работы через QWebChannel
        js = """
        if (typeof qt !== 'undefined') {
            new QWebChannel(qt.webChannelTransport, function(channel) {
                window.pyBridge = channel.objects.pyBridge;
            });
        }
        """
        self.web_view.page().runJavaScript(js)

    # ── Взаимодействие HTML → плеер ──
    def _on_seek_from_html(self, frame_idx, timestamp):
        """Плеер прыгает на таймкод при клике по карточке в HTML."""
        self.player_thread.seek(frame_idx)
        self._log(f"🎯 Прыжок к кадру {frame_idx} ({timestamp})")

    def _on_save_selected(self, selected_json):
        """Сохраняет выбранные моменты в папку output/<video>_selected_<ts>/."""
        try:
            data = json.loads(selected_json)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_stem = Path(self.video_path).stem if self.video_path else "video"
            out_dir = Path("output") / f"{video_stem}_selected_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Сохраняем JPEG выбранных + gallery.html
            for i, item in enumerate(data.get("items", [])):
                b64 = item.get("base64", "")
                if b64:
                    import base64
                    img_data = base64.b64decode(b64.split(",")[-1])
                    with open(out_dir / f"moment_{i + 1}.jpg", "wb") as f:
                        f.write(img_data)

            # gallery.html (self-contained)
            html_content = data.get("gallery_html", "")
            if html_content:
                with open(out_dir / "gallery.html", "w", encoding="utf-8") as f:
                    f.write(html_content)

            self._log(f"💾 Сохранено: {out_dir}")
            QMessageBox.information(self, "Сохранено", f"Выбранные моменты сохранены:\n{out_dir}")
        except Exception as e:
            self._log(f"❌ Ошибка сохранения: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    # ── Пакет ──
    def _on_batch(self):
        """Пакетная обработка папки."""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с видео")
        if not folder:
            return
        self.batch_btn.setEnabled(False)
        threading.Thread(target=self._run_batch, args=(folder,), daemon=True).start()

    def _run_batch(self, folder):
        """Пакетный анализ (переиспользует логику из Мини)."""
        exts = (".mp4", ".avi", ".mov", ".mkv")
        try:
            videos = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(exts)
            ])
        except Exception as e:
            self._log(f"❌ Ошибка: {e}")
            return

        if not videos:
            self._log("⚠️ В папке нет видеофайлов")
            return

        n = len(videos)
        self._log(f"📁 Пакет: {n} видео")
        total_moments = 0

        for vi, vpath in enumerate(videos):
            self._log(f"▶️ Видео {vi + 1}/{n}: {os.path.basename(vpath)}")

            def batch_progress(percent, ts, objects, extra_log=None, eta_sec=None,
                               _vi=vi, _n=n):
                overall = (_vi + min(percent, 100.0) / 100.0) / _n * 100.0
                def _update():
                    self.progress.setValue(int(overall))
                    self.progress_label.setText(f"{int(overall)}%")
                QTimer.singleShot(0, _update)

            try:
                results = self.detector.analyze_video(
                    vpath,
                    drone_mode=self.drone_check.isChecked(),
                    confidence=0.35,
                    progress_callback=batch_progress,
                )
            except Exception as e:
                self._log(f"❌ Ошибка в {os.path.basename(vpath)}: {e}")
                results = []

            total_moments += len(results)
            self._log(f"  ✅ {os.path.basename(vpath)}: {len(results)} моментов")

            # HTML для каждого видео
            try:
                from core.report_html import generate_html
                html_path = generate_html(
                    vpath, results,
                    report_data={
                        "model": self.detector.model_name,
                        "hardware": self.detector.hw,
                        "created_at": datetime.datetime.now().isoformat(),
                        "moments_count": len(results),
                    },
                )
                self._log(f"  📄 HTML: {html_path}")
            except Exception as e:
                self._log(f"  ⚠️ HTML не создан: {e}")

        self._log(f"✅ Пакет завершён: {n} видео, {total_moments} моментов")
        QTimer.singleShot(0, lambda: self.batch_btn.setEnabled(True))

    # ── Открыть HTML во внешнем браузере ──
    def _on_open_html_external(self):
        """Открывает последний HTML во внешнем браузере."""
        import webbrowser
        if self.last_html_path and os.path.exists(self.last_html_path):
            webbrowser.open(self.last_html_path)
        else:
            # Диалог выбора
            path, _ = QFileDialog.getOpenFileName(
                self, "Выберите HTML", "output", "HTML (*.html)")
            if path:
                webbrowser.open(path)

    def closeEvent(self, event):
        """Останавливает поток плеера при закрытии."""
        self.player_thread.stop()
        event.accept()


def launch():
    """Точка входа PRO-версии."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MuraveiVision PRO")
    window = ProMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()