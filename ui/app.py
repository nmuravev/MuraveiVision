"""GUI приложения MuraveiVision (CustomTkinter)."""
import os
import json
import glob
import threading
import datetime

import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from core.paths import app_root, models_dir, output_dir


def open_in_browser(path):
    """Открывает файл в браузере/программе по умолчанию через os.startfile.

    Модуль webbrowser НЕ использовать — фатальный краш GIL при загруженном PySide6.
    """
    try:
        os.startfile(os.path.abspath(path))
    except Exception as e:
        messagebox.showwarning("Браузер", f"Не удалось открыть файл:\n{e}")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── AMOLED-палитра ──
_C_BG = "#000000"        # окно и фреймы
_C_CARD = "#0D0D0D"     # карточки
_C_BORDER = "#1E1E1E"   # рамки
_C_ACCENT = "#00E5FF"   # кнопки
_C_ACCENT_HOVER = "#00B8D4"
_C_TEXT = "#E0E0E0"     # основной текст
_C_TEXT_SEC = "#8A8A8A" # вторичный
_C_SUCCESS = "#00FF88"
_C_ERROR = "#FF3B30"
_C_FONT_MONO = ("Consolas", 12)

# ── Галерея удалена (п.А1): вместо неё — HTML-отчёт в браузере ──
# Функции open_gallery, _show_full_frame, _draw_boxes, _frame_to_ctkimage,
# _extract_frame, _get_color убраны — HTML v2 заменяет галерею полностью.


def launch():
    """Точка входа GUI."""
    from core import hardware, backend
    from core.detector import MilitaryDetector
    from core.classes import MILITARY_CLASSES
    from core import nvidia_client

    app = ctk.CTk()
    app.title("🎯 MuraveiVision — Military Object Detector")
    app.geometry("1200x800")
    app.configure(fg_color=_C_BG)

    detector = MilitaryDetector()

    # Заголовок
    ctk.CTkLabel(
        app,
        text="🎯 MuraveiVision",
        font=ctk.CTkFont(size=24, weight="bold"),
        text_color=_C_ACCENT,
    ).pack(pady=15)

    # Информация о железе
    hw = hardware.detect_all()
    backend_str = backend.describe_backend(hw["gpu"], hw["cpu"])
    ctk.CTkLabel(
        app,
        text=f"⚙️ Backend: {backend_str}",
        font=ctk.CTkFont(size=14, weight="bold"),
        text_color=_C_TEXT_SEC,
    ).pack(pady=5)


    # ── Панель выбора модели (п.2: combobox + добавить) ──
    model_frame = ctk.CTkFrame(app, fg_color=_C_CARD, border_color=_C_BORDER, border_width=1)
    model_frame.pack(padx=20, pady=(0, 5), fill="x")

    ctk.CTkLabel(
        model_frame, text="🤖 Модель:",
        font=ctk.CTkFont(size=13, weight="bold"), text_color=_C_TEXT_SEC,
    ).pack(side="left", padx=(10, 5), pady=8)

    # Список моделей из models/
    def list_models():
        mdir = models_dir()
        if not mdir.exists():
            return []
        return sorted([f.name for f in mdir.glob("*.onnx")])

    model_var = ctk.StringVar(value=detector.model_name)
    model_combo = ctk.CTkComboBox(
        model_frame, variable=model_var, values=list_models(),
        width=350, fg_color="#000000", border_color=_C_BORDER,
        button_color=_C_ACCENT, button_hover_color=_C_ACCENT_HOVER,
        text_color=_C_TEXT, dropdown_fg_color=_C_CARD,
    )
    model_combo.pack(side="left", padx=5, pady=8)

    # Кнопка "➕ Добавить модель"
    def add_model():
        """askopenfilename(.onnx) → копировать в models/ → обновить список → выбрать."""
        src = filedialog.askopenfilename(
            title="Выберите ONNX-модель",
            filetypes=[("ONNX-модели", "*.onnx"), ("Все файлы", "*.*")],
        )
        if not src:
            return
        import shutil
        dst = models_dir() / os.path.basename(src)
        try:
            shutil.copy2(src, dst)
            log_msg(f"📥 Модель скопирована: {dst.name}")
        except Exception as e:
            messagebox.showerror("Ошибка", "Не удалось скопировать модель: " + str(e))
            return
        # Обновляем список
        model_combo.configure(values=list_models())
        # Выбираем добавленную
        model_var.set(dst.name)
        on_model_change(dst.name)

    ctk.CTkButton(
        model_frame, text="➕ Добавить модель", command=add_model,
        fg_color=_C_ACCENT, hover_color=_C_ACCENT_HOVER, text_color="#000000",
        width=180, height=32,
    ).pack(side="left", padx=5, pady=8)

    # ── Запись MODEL_OVERRIDE в .env ──
    def write_model_override(model_name):
        """Записывает MODEL_OVERRIDE=<имя> в .env, не трогая остальные строки."""
        env_path = str(app_root() / ".env")
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as fenv:
                lines = fenv.readlines()
        for i, line in enumerate(lines):
            if line.startswith("MODEL_OVERRIDE="):
                lines[i] = "MODEL_OVERRIDE=" + model_name + "\n"
                found = True
                break
        if not found:
            lines.append("MODEL_OVERRIDE=" + model_name + "\n")
        with open(env_path, "w", encoding="utf-8") as fenv:
            fenv.writelines(lines)
        os.environ["MODEL_OVERRIDE"] = model_name

    # ── Смена модели: пересоздать детектор в фоновом потоке ──
    def on_model_change(new_name):
        """Пересоздаёт детектор с выбранной моделью в фоновом потоке.

        ВАЖНО: НЕ перезапускает процесс/окно — только пересоздаёт детектор
        в фоновом потоке внутри текущего окна. Второе окно не появляется.
        """
        if not new_name:
            return
        # Проверяем, что файл существует
        if not (models_dir() / new_name).exists():
            log_msg(f"⚠️ Модель не найдена: {new_name}")
            return
        # Записываем в .env
        write_model_override(new_name)
        log_msg("🔄 Загрузка модели...")
        # start_btn может быть ещё не создан (ранний вызов) — защищаемся
        if start_btn is not None:
            start_btn.configure(state="disabled", text="🔄 Загрузка модели...")

        def _reload():
            nonlocal detector
            try:
                os.environ["MODEL_OVERRIDE"] = new_name
                from core.detector import MilitaryDetector
                new_det = MilitaryDetector()
                detector = new_det
                app.after(0, lambda: log_msg(f"✅ Модель: {detector.model_name}"))
                app.after(0, lambda: log_msg(f"🖥️ Провайдеры: {', '.join(detector.providers)}"))
            except Exception as e:
                app.after(0, lambda err=e: log_msg(f"❌ Ошибка загрузки модели: {err}"))
            finally:
                app.after(0, lambda: start_btn.configure(state="normal", text="🚀 Начать анализ"))

        threading.Thread(target=_reload, daemon=True).start()

    # Привязываем смену модели в combobox
    model_combo.configure(command=on_model_change)

    # Фрейм управления
    ctrl = ctk.CTkFrame(app, fg_color=_C_BG, border_color=_C_BORDER, border_width=1)
    ctrl.pack(padx=20, pady=10, fill="x")

    video_path_var = ctk.StringVar(value="")
    video_label = ctk.CTkLabel(ctrl, text="📁 Видео не выбрано", text_color=_C_TEXT_SEC)
    video_label.grid(row=0, column=1, padx=10, sticky="w")

    def choose_video():
        p = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv")],
        )
        if p:
            video_path_var.set(p)
            video_label.configure(text=os.path.basename(p), text_color=_C_TEXT)

    ctk.CTkButton(
        ctrl, text="📹 Выбрать видео", command=choose_video, width=180,
        fg_color=_C_ACCENT, hover_color=_C_ACCENT_HOVER, text_color="#000000",
    ).grid(row=0, column=0, padx=10, pady=10)

    drone_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(ctrl, text="🎯 🚁 Режим дрона", variable=drone_var,
                    text_color=_C_TEXT).grid(
        row=1, column=0, padx=10, pady=5, sticky="w"
    )

    # Чекбокс слоя 4 (облачная обработка данных)
    cloud_var = ctk.BooleanVar(value=False)
    cloud_cb = ctk.CTkCheckBox(
        ctrl, text="☁️ Облачная обработка данных", variable=cloud_var,
        text_color=_C_TEXT,
    )
    cloud_cb.grid(row=2, column=0, padx=10, pady=5, sticky="w")

    # Контейнер для поля ключа и предупреждения (показывается при галке)
    cloud_frame = ctk.CTkFrame(ctrl, fg_color="#0D0D0D", border_color="#1E1E1E", border_width=1)
    # Изначально скрыт
    cloud_frame.grid_forget()

    # Поле API-ключа NVIDIA
    ctk.CTkLabel(
        cloud_frame, text="🔑 API-ключ NVIDIA:",
        font=ctk.CTkFont(size=12, weight="bold"), text_color="#E0E0E0",
    ).pack(padx=10, pady=(10, 2), anchor="w")

    api_key_var = ctk.StringVar(value=os.getenv("NVIDIA_API_KEY", ""))
    # Маскировка ключа (п.3): show="*" по умолчанию, toggle через 👁
    api_key_entry = ctk.CTkEntry(
        cloud_frame, textvariable=api_key_var, width=400,
        placeholder_text="nvapi-...", fg_color="#000000", border_color="#1E1E1E",
        text_color="#E0E0E0", show="*",
    )
    api_key_entry.pack(padx=10, pady=2, side="left", fill="x", expand=True)

    # Иконка 👁 — показать/скрыть ключ
    def toggle_key_visibility():
        if api_key_entry.cget("show") == "*":
            api_key_entry.configure(show="")
            key_toggle_btn.configure(text="🙈")
        else:
            api_key_entry.configure(show="*")
            key_toggle_btn.configure(text="👁")

    key_toggle_btn = ctk.CTkButton(
        cloud_frame, text="👁", command=toggle_key_visibility,
        fg_color="#1E1E1E", hover_color="#2A2A2A", text_color="#00E5FF",
        width=45, height=32,
    )
    key_toggle_btn.pack(padx=(5, 10), pady=2, side="left")

    def save_api_key():
        """Сохраняет/обновляет NVIDIA_API_KEY в .env, не трогая остальные строки."""
        key = api_key_var.get().strip()
        env_path = str(app_root() / ".env")
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("NVIDIA_API_KEY="):
                lines[i] = f"NVIDIA_API_KEY={key}\n"
                found = True
                break
        if not found:
            lines.append(f"NVIDIA_API_KEY={key}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        # Обновляем окружение текущего процесса
        os.environ["NVIDIA_API_KEY"] = key
        log_msg("🔑 Ключ сохранён")

    ctk.CTkButton(
        cloud_frame, text="🔑 Сохранить", command=save_api_key,
        fg_color="#00E5FF", hover_color="#00B8D4", text_color="#000000",
        width=160, height=32,
    ).pack(padx=10, pady=5, anchor="w")

    # Красное жирное предупреждение
    ctk.CTkLabel(
        cloud_frame,
        text="⚠️ НЕОБХОДИМО ОТКЛЮЧИТЬ ГЕОЛОКАЦИЮ И ВКЛЮЧИТЬ ВНЕШНИЙ ПРОКСИ",
        font=ctk.CTkFont(size=12, weight="bold"), text_color="red",
        wraplength=500, justify="left",
    ).pack(padx=10, pady=(5, 10), anchor="w")

    def toggle_cloud_frame():
        """Показывает/скрывает блок поля ключа при установке галки."""
        if cloud_var.get():
            cloud_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        else:
            cloud_frame.grid_forget()

    cloud_cb.configure(command=toggle_cloud_frame)

    conf_label = ctk.CTkLabel(ctrl, text="🎯 Чувствительность: 0.35",
                              text_color=_C_TEXT)
    conf_label.grid(row=1, column=1, sticky="w")

    def update_conf(v):
        conf_label.configure(text=f"Чувствительность: {float(v):.2f}")

    conf_slider = ctk.CTkSlider(
        ctrl, from_=0.05, to=0.95, number_of_steps=18, command=update_conf
    )
    conf_slider.set(0.35)
    conf_slider.grid(row=1, column=2, padx=10, sticky="ew")

    # Подпись над прогресс-баром (п.4: "45% • осталось 1м 32с")
    progress_label = ctk.CTkLabel(app, text="0%", text_color=_C_TEXT_SEC,
                                  font=ctk.CTkFont(_C_FONT_MONO[0], 11))
    progress_label.pack(padx=20, anchor="w")

    # Прогресс и лог
    progress = ctk.CTkProgressBar(app, progress_color=_C_ACCENT, fg_color=_C_BORDER)
    progress.pack(padx=20, fill="x")
    progress.set(0)

    log = ctk.CTkTextbox(app, height=400, fg_color=_C_CARD, border_color=_C_BORDER,
                         border_width=1, text_color=_C_TEXT,
                         font=ctk.CTkFont(_C_FONT_MONO[0], _C_FONT_MONO[1]))
    log.pack(padx=20, pady=10, fill="both", expand=True)

    def log_msg(msg):
        log.insert("end", msg + "\n")
        log.see("end")

    log_msg("✅ Приложение готово к работе")
    log_msg(f"📋 Классов в словаре: {len(MILITARY_CLASSES)}")
    log_msg(f"🤖 Модель: {detector.model_name}")
    log_msg(f"🖥️ Провайдеры: {', '.join(detector.providers)}")

    # Проверка прокси-политики (п.Б4): без NVIDIA_PROXY облако не вызывается
    _nvidia_proxy = os.getenv("NVIDIA_PROXY", "")
    if not _nvidia_proxy:
        log_msg("☁️ Облако отключено: политика — только через прокси")
    elif nvidia_client.is_available():
        log_msg("☁️ Слой 4 (NVIDIA Vision) доступен")
    else:
        log_msg("☁️ Слой 4 недоступен офлайн — работаем на локальных слоях")

    start_btn = None

    # Переменные для повторного открытия галереи
    last_video_path = {"value": None}
    last_moments = {"value": []}
    # Переменные для путей отчётов (HTML/JSON) — для кнопки "📄 Открыть HTML"
    last_html_path = {"value": None}
    last_report_path = {"value": None}

    def on_progress(percent, timestamp_str, objects, extra_log=None, eta_sec=None):
        """Колбэк прогресса из детектора (вызывается из рабочего потока).

        Аргументы:
          percent       — процент выполнения (0..100);
          timestamp_str — таймкод текущего кадра;
          objects       — список найденных объектов;
          extra_log     — опциональное сообщение для лога (напр. о стоп-кадрах);
          eta_sec       — опциональная оценка оставшегося времени (сек).
        """
        def _update():
            progress.set(min(percent / 100.0, 1.0))
            # ETA: форматируем "1м 32с"
            eta_str = ""
            if eta_sec is not None and eta_sec >= 0:
                mins = eta_sec // 60
                secs = eta_sec % 60
                eta_str = f" • осталось {mins}м {secs}с"
                # Подпись над прогресс-баром
                progress_label.configure(text=f"{int(percent)}%{eta_str}")
            if extra_log:
                log_msg(extra_log)
            elif objects:
                classes = [o["class"] for o in objects]
                log_msg(f"⏱️ [{int(percent)}%]{eta_str} | {timestamp_str} | "
                        f"{len(objects)} объектов: {', '.join(classes[:5])}")
        app.after(0, _update)

    def _run_cloud_analysis(video_path, results):
        """Запускает облачную проверку до 5 кропов с наименьшей confidence.

        Возвращает dict {frame_idx: str} — ответы слоя 4.
        """
        if not nvidia_client.is_available():
            app.after(0, lambda: log_msg("☁️ Слой 4 недоступен офлайн — работаем на локальных слоях"))
            return {}

        # Собираем все объекты со всех моментов, сортируем по confidence (возрастание)
        all_objs = []
        for moment in results:
            for obj in moment["objects"]:
                all_objs.append((moment["frame"], obj, moment))

        # Берём до 5 кропов с наименьшей confidence
        all_objs.sort(key=lambda x: x[1]["confidence"])
        to_check = all_objs[:5]

        if not to_check:
            return {}

        app.after(0, lambda: log_msg(f"☁️ Облачная проверка {len(to_check)} подозрительных кропов..."))

        annotations = {}
        cap = cv2.VideoCapture(video_path)

        for frame_idx, obj, moment in to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            app.after(0, lambda o=obj: log_msg(
                f"  ☁️ Проверка: {o['class']} (conf={o['confidence']:.2f})..."))

            answer = nvidia_client.analyze_crop(frame, obj["bbox"])
            if answer:
                annotations[frame_idx] = answer
                app.after(0, lambda a=answer, o=obj: log_msg(
                    f"  ✅ {o['class']}: {a[:100]}..."))
            else:
                app.after(0, lambda o=obj: log_msg(
                    f"  ⚠️ {o['class']}: нет ответа от слоя 4"))

        cap.release()
        return annotations

    def run_analysis():
        path = video_path_var.get()
        if not path:
            messagebox.showwarning("Ошибка", "Сначала выберите видео!")
            start_btn.configure(state="normal", text="🚀 Начать анализ")
            return
        try:
            # Сброс всех счётчиков прогресса/ETA в начале КАЖДОГО запуска
            progress.set(0)
            progress_label.configure(text="0%")
            log_msg(f"▶️ Запуск анализа: {os.path.basename(path)}")
            results = detector.analyze_video(
                path,
                drone_mode=drone_var.get(),
                confidence=conf_slider.get(),
                progress_callback=on_progress,
            )
            app.after(0, lambda: progress.set(1.0))
            app.after(0, lambda: log_msg(f"✅ Найдено моментов: {len(results)}"))
            if detector.last_report_path:
                app.after(0, lambda: log_msg(f"💾 Отчёт сохранён: {detector.last_report_path}"))

            # Сохраняем результаты для повторного открытия галереи
            last_video_path["value"] = path
            last_moments["value"] = results
            # Сохраняем путь JSON-отчёта
            if detector.last_report_path:
                last_report_path["value"] = detector.last_report_path

            # Облачная проверка (слой 4), если включена
            cloud_annotations = {}
            if cloud_var.get() and results:
                cloud_annotations = _run_cloud_analysis(path, results)

            # Генерация HTML-отчёта (п.3) — самодостаточный файл с base64-миниатюрами
            try:
                from core.report_html import generate_html
                settings = {
                    "drone_mode": drone_var.get(),
                    "confidence": round(conf_slider.get(), 2),
                    "frame_step": detector.frame_step,
                    "drone_frame_step": detector.drone_frame_step,
                }
                report_data = {
                    "model": detector.model_name,
                    "hardware": detector.hw,
                    "created_at": datetime.datetime.now().isoformat(),
                    "moments_count": len(results),
                }
                html_path = generate_html(
                    path, results,
                    report_data=report_data,
                    settings=settings,
                    cloud_annotations=cloud_annotations,
                )
                last_html_path["value"] = html_path
                app.after(0, lambda p=html_path: log_msg(f"📄 HTML-отчёт: {p}"))
            except Exception as e:
                app.after(0, lambda err=e: log_msg(f"⚠️ HTML-отчёт не создан: {err}"))

            # Вместо галереи (п.А1) — открываем HTML-отчёт в браузере (БАГ 4: убрано ложное "Ничего не найдено")
            if last_html_path["value"]:
                app.after(0, lambda p=last_html_path["value"]: open_in_browser(p))
            elif not results:
                app.after(0, lambda: log_msg("Ничего не найдено"))

            app.after(
                0, lambda: messagebox.showinfo("Готово", f"Найдено моментов: {len(results)}")
            )
        except Exception as e:
            app.after(0, lambda: log_msg(f"❌ Ошибка: {e}"))
            app.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            app.after(
                0,
                lambda: start_btn.configure(state="normal", text="🚀 Начать анализ"),
            )

    def start_analysis():
        start_btn.configure(state="disabled", text="⏳ Анализ...")
        threading.Thread(target=run_analysis, daemon=True).start()

    # ── Пакетная обработка папки (п.1) ──
    def run_batch_analysis(folder):
        """Анализирует все видео в папке одним экземпляром MilitaryDetector.

        Для каждого видео:
          - отчёт сохраняется как обычно (detector.analyze_video);
          - в output/ пишется batch_summary_<ts>.json;
          - в конце — общий HTML-отчёт по всей папке.
        Общий прогресс = (индекс видео + процент внутри видео) / N.
        """
        # Расширения видео
        exts = (".mp4", ".avi", ".mov", ".mkv")
        try:
            videos = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(exts)
            ])
        except Exception as e:
            app.after(0, lambda err=e: log_msg(f"❌ Ошибка чтения папки: {err}"))
            return

        if not videos:
            app.after(0, lambda: log_msg("⚠️ В папке нет видеофайлов"))
            return

        n = len(videos)
        app.after(0, lambda: log_msg(f"📁 Пакет: найдено {n} видео в {folder}"))

        # Сводка по пакету
        batch_summary = []
        total_moments = 0
        all_batch_moments = []  # для общего HTML: [{video, moments, report_path}]

        try:
            for vi, vpath in enumerate(videos):
                vname = os.path.basename(vpath)
                app.after(0, lambda i=vi, nm=vname: log_msg(
                    f"▶️ Видео {i + 1}/{n}: {nm}"))

                # Сброс прогресса/ETA в начале каждого видео пакета
                app.after(0, lambda i=vi: (
                    progress.set(0),
                    progress_label.configure(text=f"Видео {i + 1}/{n}: 0%"),
                ))

                # Колбэк прогресса с учётом позиции в пакете + живой ETA
                def batch_progress(percent, timestamp_str, objects, extra_log=None,
                                   eta_sec=None, _vi=vi, _n=n):
                    """Колбэк прогресса для пакетного режима.

                    Общий прогресс = (индекс видео + процент внутри видео) / N.
                    ETA учитывает только текущее видео (живой прогресс, не 100%).
                    """
                    overall = (_vi + min(percent, 100.0) / 100.0) / _n * 100.0

                    def _update():
                        progress.set(min(overall / 100.0, 1.0))
                        # Живой ETA для текущего видео
                        eta_str = ""
                        if eta_sec is not None and eta_sec >= 0:
                            mins = eta_sec // 60
                            secs = eta_sec % 60
                            eta_str = f" • осталось {mins}м {secs}с"
                        progress_label.configure(
                            text=f"Видео {_vi + 1}/{_n}: {int(percent)}%{eta_str}"
                        )
                        if extra_log:
                            log_msg(extra_log)
                        elif objects:
                            classes = [o["class"] for o in objects]
                            log_msg(
                                f"  ⏱️ {timestamp_str} | {len(objects)} объектов: "
                                f"{', '.join(classes[:5])}"
                            )
                    app.after(0, _update)

                # Анализ одного видео (модель НЕ пересоздаём — используем тот же detector)
                try:
                    results = detector.analyze_video(
                        vpath,
                        drone_mode=drone_var.get(),
                        confidence=conf_slider.get(),
                        progress_callback=batch_progress,
                    )
                except Exception as e:
                    app.after(0, lambda err=e, nm=vname: log_msg(
                        f"❌ Ошибка в {nm}: {err}"))
                    results = []

                app.after(0, lambda r=results, nm=vname: log_msg(
                    f"  ✅ {nm}: {len(r)} моментов"))
                total_moments += len(results)

                # Сводка классов для этого видео
                classes_summary = {}
                for m in results:
                    for o in m.get("objects", []):
                        cn = o.get("class", "?")
                        classes_summary[cn] = classes_summary.get(cn, 0) + 1

                report_path = getattr(detector, "last_report_path", None)
                entry = {
                    "файл": vname,
                    "моментов": len(results),
                    "классов_сводка": classes_summary,
                    "путь_отчёта": report_path,
                }
                batch_summary.append(entry)
                all_batch_moments.append({
                    "video": vpath,
                    "video_name": vname,
                    "moments": results,
                    "report_path": report_path,
                })

                # HTML-отчёт для каждого видео пакета
                try:
                    from core.report_html import generate_html
                    settings = {
                        "drone_mode": drone_var.get(),
                        "confidence": round(conf_slider.get(), 2),
                        "frame_step": detector.frame_step,
                        "drone_frame_step": detector.drone_frame_step,
                        "batch": f"{vi + 1}/{n}",
                    }
                    report_data = {
                        "model": detector.model_name,
                        "hardware": detector.hw,
                        "created_at": datetime.datetime.now().isoformat(),
                        "moments_count": len(results),
                    }
                    html_path = generate_html(
                        vpath, results,
                        report_data=report_data,
                        settings=settings,
                    )
                    app.after(0, lambda p=html_path, nm=vname: log_msg(
                        f"  📄 HTML-отчёт ({nm}): {p}"))
                except Exception as e:
                    app.after(0, lambda err=e, nm=vname: log_msg(
                        f"  ⚠️ HTML для {nm} не создан: {err}"))

            # Сохраняем batch_summary_<ts>.json
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_path = str(output_dir() / f"batch_summary_{ts}.json")
                output_dir().mkdir(parents=True, exist_ok=True)
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "folder": folder,
                        "videos_count": n,
                        "total_moments": total_moments,
                        "items": batch_summary,
                    }, f, ensure_ascii=False, indent=2)
                app.after(0, lambda p=summary_path: log_msg(
                    f"📋 Сводка пакета: {p}"))
            except Exception as e:
                app.after(0, lambda err=e: log_msg(f"⚠️ Сводка не сохранена: {err}"))

            # Общий HTML-отчёт по всей папке
            try:
                from core.report_html import generate_html
                # Используем первое видео как "основное" для шапки,
                # но в карточки включаем моменты всех видео с пометкой источника.
                # Простейший подход: генерируем HTML для каждого видео уже сделан выше,
                # а здесь создаём индексный HTML-отчёт со ссылками на все видео.
                # Однако ТЗ требует "общий HTML-отчёт по всей папке".
                # Создаём объединённый отчёт: шапка пакета + карточки всех моментов.
                _generate_batch_html(folder, all_batch_moments, detector, n,
                                     total_moments)
            except Exception as e:
                app.after(0, lambda err=e: log_msg(
                    f"⚠️ Общий HTML-отчёт не создан: {err}"))

            app.after(0, lambda: log_msg(
                f"✅ Пакет завершён: {n} видео, {total_moments} моментов"))
            app.after(0, lambda: progress.set(1.0))
            app.after(0, lambda: messagebox.showinfo(
                "Пакет завершён",
                f"Обработано видео: {n}\nВсего моментов: {total_moments}",
            ))
        finally:
            app.after(
                0,
                lambda: start_btn.configure(state="normal", text="🚀 Начать анализ"),
            )

    def _generate_batch_html(folder, all_batch_moments, det, n, total_moments):
        """Создаёт общий HTML-отчёт по всей папке (п.1 + п.3).

        Шапка: папка, дата, железо, модель, кол-во видео/моментов.
        Карточки: все моменты всех видео (с пометкой источника), макс. 200 миниатюр.
        """
        from core.report_html import generate_html, _draw_boxes, _frame_to_base64
        import cv2 as _cv2

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = str(output_dir() / f"batch_report_{ts}.html")

        # Собираем все моменты с пометкой источника
        all_moments = []
        for item in all_batch_moments:
            for m in item["moments"]:
                # Копируем момент и добавляем имя видео
                mm = dict(m)
                mm["_video_name"] = item["video_name"]
                mm["_video_path"] = item["video"]
                all_moments.append(mm)

        # Ограничиваем 200 миниатюр
        display = all_moments[:200]

        # Шапка
        hw = det.hw
        if isinstance(hw, dict):
            gpu = hw.get("gpu")
            cpu = hw.get("cpu", {})
            if gpu:
                hw_str = f"GPU: {gpu.get('name', '')} ({gpu.get('vram_mb', 0) // 1024} GB)"
            else:
                hw_str = f"CPU: {cpu.get('brand', '')}"
        else:
            hw_str = str(hw)

        # Карточки с пометкой источника
        cards_html = ""
        for i, moment in enumerate(display):
            ts_m = moment.get("timestamp", "")
            vname = moment.get("_video_name", "")
            objects = moment.get("objects", [])
            obj_lines = ""
            for o in objects:
                obj_lines += (
                    f"<div class='obj-row'>"
                    f"<span class='obj-class'>{o.get('class', '')}</span>"
                    f"<span class='obj-conf'>{o.get('confidence', 0):.2f}</span></div>"
                )
            cards_html += f"""
            <div class='card' data-img-id='{i}'>
                <div class='card-thumb'><img class='thumb' alt='moment {i}' /></div>
                <div class='card-info'>
                    <div class='card-ts mono'>⏱️ {ts_m}</div>
                    <div class='card-src mono'>📁 {vname}</div>
                    <div class='card-objs'>{obj_lines}</div>
                </div>
            </div>
            """

        css = f"""
        <style>
        * {{ box-sizing: border-box; }}
        body {{ margin:0; padding:20px; background:#000000; color:#E0E0E0;
               font-family:'Segoe UI','Roboto',sans-serif; }}
        .mono {{ font-family:'Consolas','Courier New',monospace; }}
        .header {{ background:#0D0D0D; border:1px solid #1E1E1E; border-radius:12px;
                   padding:20px; margin-bottom:20px; }}
        .header h1 {{ color:#00E5FF; margin:0 0 15px 0; font-size:24px; }}
        .meta {{ display:flex; flex-direction:column; gap:6px; }}
        .kv-row {{ display:flex; gap:12px; font-size:14px; }}
        .kv-key {{ color:#8A8A8A; min-width:120px; }}
        .kv-val {{ color:#E0E0E0; }}
        .card {{ display:flex; gap:16px; background:#0D0D0D; border:1px solid #1E1E1E;
                 border-radius:10px; padding:12px; margin-bottom:14px; }}
        .card-thumb img {{ width:480px; border-radius:6px; display:block; }}
        .card-info {{ flex:1; }}
        .card-ts {{ color:#00E5FF; font-size:16px; font-weight:bold; margin-bottom:4px; }}
        .card-src {{ color:#8A8A8A; font-size:12px; margin-bottom:8px; }}
        .card-objs {{ display:flex; flex-direction:column; gap:4px; }}
        .obj-row {{ display:flex; justify-content:space-between; max-width:300px; }}
        .obj-class {{ color:#E0E0E0; }}
        .obj-conf {{ color:#00FF88; font-family:'Consolas',monospace; }}
        .footer {{ text-align:center; color:#8A8A8A; margin-top:30px; font-size:12px; }}
        </style>
        """

        html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MuraveiVision — Пакетный отчёт — {ts}</title>
{css}</head><body>
<header class="header">
    <h1>🎯 MuraveiVision — Пакетный отчёт</h1>
    <div class="meta">
        <div class="kv-row"><span class="kv-key">Папка</span>
            <span class="kv-val mono">{folder}</span></div>
        <div class="kv-row"><span class="kv-key">Дата</span>
            <span class="kv-val">{datetime.datetime.now().isoformat()}</span></div>
        <div class="kv-row"><span class="kv-key">Железо</span>
            <span class="kv-val">{hw_str}</span></div>
        <div class="kv-row"><span class="kv-key">Модель</span>
            <span class="kv-val mono">{det.model_name}</span></div>
        <div class="kv-row"><span class="kv-key">Видео</span>
            <span class="kv-val" style="color:#00FF88">{n}</span></div>
        <div class="kv-row"><span class="kv-key">Всего моментов</span>
            <span class="kv-val" style="color:#00FF88">{total_moments}</span></div>
    </div>
</header>
<main>
{cards_html}
</main>
<div class="footer">Сгенерировано MuraveiVision · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body></html>
"""

        # Встраиваем миниатюры (по источнику видео)
        # Группируем по видео, чтобы открывать каждый cap один раз
        by_video = {}
        for i, moment in enumerate(display):
            vp = moment.get("_video_path", "")
            by_video.setdefault(vp, []).append((i, moment))

        for vp, items in by_video.items():
            if not vp or not os.path.exists(vp):
                continue
            cap = _cv2.VideoCapture(vp)
            for i, moment in items:
                frame_idx = moment.get("frame", 0)
                cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                annotated = _draw_boxes(frame, moment.get("objects", []))
                data_url = _frame_to_base64(annotated, max_width=480, quality=70)
                if data_url:
                    placeholder = f"<img class='thumb' alt='moment {i}' />"
                    real = f"<img class='thumb' src='{data_url}' alt='moment {i}' />"
                    html = html.replace(placeholder, real, 1)
            cap.release()

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html)

        last_html_path["value"] = out_file
        app.after(0, lambda p=out_file: log_msg(f"📄 HTML-отчёт (пакет): {p}"))

    def start_batch():
        """Обработчик кнопки '📁 Папка (пакетно)'."""
        folder = filedialog.askdirectory(title="Выберите папку с видео")
        if not folder:
            return
        start_btn.configure(state="disabled", text="⏳ Пакет...")
        threading.Thread(target=run_batch_analysis, args=(folder,), daemon=True).start()

    # ── Кнопка "📂 Открыть отчёт" (п.2, БАГ 3) ──
    def open_report_click():
        """Открывает JSON-отчёт из output/, генерирует HTML и открывает в браузере.

        Миниатюры берутся:
          1) из папки output/<video_stem>_frames/ по совпадению таймкода;
          2) если папки нет — из видео (если видеофайл существует);
          3) если нет ни того ни другого — карточки без миниатюр.
        """
        # Диалог выбора JSON из output/
        initial = str(output_dir()) if output_dir().exists() else os.getcwd()
        report_file = filedialog.askopenfilename(
            title="Выберите JSON-отчёт",
            initialdir=initial,
            filetypes=[("JSON-отчёты", "*.json"), ("Все файлы", "*.*")],
        )
        if not report_file:
            return

        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать отчёт:\n{e}")
            return

        video_path = data.get("video", "")
        moments = data.get("moments", [])

        if not moments:
            messagebox.showinfo("Отчёт", "В отчёте нет моментов")
            return

        # Заполняем переменные
        last_video_path["value"] = video_path
        last_moments["value"] = moments
        last_report_path["value"] = report_file

        log_msg(f"📂 Открыт отчёт: {report_file}")
        log_msg(f"   Видео: {video_path}")
        log_msg(f"   Моментов: {len(moments)}")

        # ── БАГ 3: подготовка миниатюр ──
        # 1) Пытаемся взять из папки output/<video_stem>_frames/ по совпадению таймкода
        video_stem = os.path.splitext(os.path.basename(video_path))[0] if video_path else ""
        frames_dir = str(output_dir() / f"{video_stem}_frames") if video_stem else None
        frames_available = frames_dir and os.path.isdir(frames_dir)
        video_available = video_path and os.path.exists(video_path)

        if not frames_available and not video_available:
            log_msg("⚠️ Нет ни папки кадров, ни видео — карточки будут без миниатюр")

        # Генерируем HTML из загруженного отчёта (БАГ 3: без ошибок в логе)
        try:
            from core.report_html import generate_html
            report_data = {
                "model": data.get("model", ""),
                "hardware": data.get("hardware", {}),
                "created_at": data.get("created_at", ""),
                "moments_count": len(moments),
            }
            # Если видео недоступно — передаём пустой путь, generate_html создаст карточки без миниатюр
            html_video_path = video_path if video_available else ""
            html_path = generate_html(html_video_path, moments, report_data=report_data)
            last_html_path["value"] = html_path
            open_in_browser(html_path)
            log_msg(f"📄 HTML-отчёт: {html_path}")
        except Exception as e:
            log_msg(f"⚠️ Не удалось создать HTML: {e}")
            messagebox.showwarning("HTML", f"Не удалось создать HTML:\n{e}\n\nОтчёт загружен, путь: {report_file}")

    # ── Кнопка "📄 Открыть HTML" (п.3, БАГ 2) ──
    def open_html_click():
        """Открывает последний HTML-отчёт (.html) в браузере.

        Если .html нет — генерирует HTML из последнего JSON (last_report_path).
        НИКОГДА не открывает .json как HTML.
        Если нет ни того ни другого — messagebox "Отчётов пока нет".
        """
        html_path = last_html_path["value"]

        # Если HTML есть и существует — открываем
        if html_path and os.path.exists(html_path):
            try:
                open_in_browser(html_path)
                log_msg(f"📄 Открыт HTML: {html_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть HTML:\n{e}")
            return

        # HTML нет — пытаемся сгенерировать из последнего JSON (БАГ 2)
        json_path = last_report_path["value"]
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                video_path = data.get("video", "")
                moments = data.get("moments", [])
                if not moments:
                    messagebox.showinfo("Отчётов пока нет", "В отчёте нет моментов")
                    return
                from core.report_html import generate_html
                report_data = {
                    "model": data.get("model", ""),
                    "hardware": data.get("hardware", {}),
                    "created_at": data.get("created_at", ""),
                    "moments_count": len(moments),
                }
                # Если видео недоступно — карточки без миниатюр
                html_video_path = video_path if os.path.exists(video_path) else ""
                html_path = generate_html(html_video_path, moments, report_data=report_data)
                last_html_path["value"] = html_path
                open_in_browser(html_path)
                log_msg(f"📄 HTML-отчёт (из JSON): {html_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось создать HTML из отчёта:\n{e}")
            return

        # Нет ни HTML, ни JSON
        messagebox.showinfo("Отчётов пока нет", "Сначала выполните анализ или откройте JSON-отчёт.")

    # ── Фрейм с кнопками (п.А2): [📁 Пакетная обработка][📂 Открыть отчёт]
    #    [📄 Открыть HTML][▶️ Анализ] ──
    # Кнопка "📄 Файл" убрана — дублирует "📹 Выбрать видео" выше.
    btn_frame = ctk.CTkFrame(app, fg_color=_C_BG)
    btn_frame.pack(pady=10)

    # Кнопка "📁 Пакетная обработка" — пакетная обработка папки
    batch_btn = ctk.CTkButton(
        btn_frame,
        text="📁 Пакетная обработка",
        command=start_batch,
        fg_color=_C_ACCENT,
        hover_color=_C_ACCENT_HOVER,
        text_color="#000000",
        height=45,
        width=200,
    )
    batch_btn.pack(side="left", padx=8)

    # Кнопка "📂 Открыть отчёт" — загрузка JSON-отчёта
    open_report_btn = ctk.CTkButton(
        btn_frame,
        text="📂 Открыть отчёт",
        command=open_report_click,
        fg_color=_C_ACCENT,
        hover_color=_C_ACCENT_HOVER,
        text_color="#000000",
        height=45,
        width=180,
    )
    open_report_btn.pack(side="left", padx=8)

    # Кнопка "📄 Открыть HTML" — открыть HTML-отчёт в браузере
    open_html_btn = ctk.CTkButton(
        btn_frame,
        text="📄 Открыть HTML",
        command=open_html_click,
        fg_color=_C_ACCENT,
        hover_color=_C_ACCENT_HOVER,
        text_color="#000000",
        height=45,
        width=180,
    )
    open_html_btn.pack(side="left", padx=8)

    # Кнопка "▶️ Анализ" — запуск одиночного анализа
    start_btn = ctk.CTkButton(
        btn_frame,
        text="▶️ Анализ",
        command=start_analysis,
        fg_color=_C_SUCCESS,
        hover_color="#00CC6E",
        text_color="#000000",
        height=45,
        width=160,
    )
    start_btn.pack(side="left", padx=8)

    app.mainloop()


if __name__ == "__main__":
    launch()
