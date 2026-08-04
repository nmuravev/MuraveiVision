"""GUI приложения MuraveiVision (CustomTkinter)."""
import os
import threading
import datetime

import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Цвета для bbox разных классов
_COLORS = [
    "#ff4444", "#44ff44", "#4444ff", "#ffff44", "#ff44ff",
    "#44ffff", "#ff8844", "#88ff44", "#4488ff", "#ff4488",
]


def _get_color(class_name):
    """Детерминированный цвет для класса по хэшу имени."""
    return _COLORS[hash(class_name) % len(_COLORS)]


def _draw_boxes(frame, objects):
    """Рисует рамки bbox и имена классов на кадре."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    thickness = max(1, int(min(h, w) / 300))

    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        color = _get_color(obj["class"])
        # BGR цвет для cv2
        b, g, r = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (b, g, r), thickness)

        # Подпись
        label = f"{obj['class']} {obj['confidence']:.2f}"
        font_scale = max(0.4, min(h, w) / 800)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), (b, g, r), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def _frame_to_ctkimage(frame, max_width=480):
    """Конвертирует BGR-кадр в CTkImage с заданной шириной."""
    h, w = frame.shape[:2]
    scale = max_width / w if w > max_width else 1.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    if scale < 1.0:
        frame = cv2.resize(frame, (new_w, new_h))

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))


def _extract_frame(video_path, frame_idx):
    """Достаёт конкретный кадр из видео по индексу."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def open_gallery(parent, video_path, moments, cloud_annotations=None):
    """Открывает окно галереи моментов.

    Аргументы:
      parent             — родительское окно;
      video_path         — путь к видео;
      moments            — список моментов [{timestamp, frame, objects}];
      cloud_annotations  — dict {frame_idx: str} — ответы слоя 4 (опционально).
    """
    if not moments:
        return

    cloud_annotations = cloud_annotations or {}

    gallery = ctk.CTkToplevel(parent)
    gallery.title(f"🎞 Галерея моментов ({len(moments)})")
    gallery.geometry("1000x700")

    # Заголовок
    ctk.CTkLabel(
        gallery,
        text=f"Найдено моментов: {len(moments)}",
        font=ctk.CTkFont(size=16, weight="bold"),
    ).pack(pady=10)

    # Прогресс генерации миниатюр
    gen_label = ctk.CTkLabel(gallery, text="🖼 Генерация галереи...", text_color="#88ccff")
    gen_label.pack(pady=5)

    # Скроллируемый контейнер
    scroll = ctk.CTkScrollableFrame(gallery)
    scroll.pack(padx=10, pady=10, fill="both", expand=True)

    # Ограничиваем до 200 моментов
    display_moments = moments[:200]
    cards = []

    def generate_thumbnails():
        """Генерирует миниатюры в отдельном потоке."""
        for i, moment in enumerate(display_moments):
            frame = _extract_frame(video_path, moment["frame"])
            if frame is None:
                continue

            # Рисуем рамки
            annotated = _draw_boxes(frame, moment["objects"])
            ctk_img = _frame_to_ctkimage(annotated, max_width=480)

            # Подпись
            obj_summary = ", ".join(
                f"{o['class']} {o['confidence']:.2f}" for o in moment["objects"][:5]
            )
            caption = f"{moment['timestamp']} — {obj_summary}"

            # Аннотация слоя 4 (если есть)
            cloud_text = cloud_annotations.get(moment["frame"], "")

            # Обновляем UI через after
            def _add_card(img=ctk_img, cap=caption, cloud=cloud_text, idx=i, fr=frame, m=moment):
                card = ctk.CTkFrame(scroll)
                card.pack(pady=5, padx=10, fill="x")

                thumb = ctk.CTkLabel(card, image=img, text="")
                thumb.image = img
                thumb.pack(side="left", padx=10, pady=10)

                info = ctk.CTkLabel(card, text=cap, font=ctk.CTkFont(size=13), wraplength=400)
                info.pack(side="left", padx=10, pady=5, anchor="n")

                if cloud:
                    cloud_label = ctk.CTkLabel(
                        card, text=f"☁️ {cloud[:200]}",
                        font=ctk.CTkFont(size=11), text_color="#ffcc88",
                        wraplength=400, justify="left",
                    )
                    cloud_label.pack(side="left", padx=10, pady=5, anchor="s")

                # Двойной клик — открыть полный кадр
                def _on_double_click(e=None, f=fr, mm=m):
                    _show_full_frame(gallery, f, mm)

                card.bind("<Double-Button-1>", _on_double_click)
                thumb.bind("<Double-Button-1>", _on_double_click)
                cards.append(card)

            gallery.after(0, _add_card)

            # Обновляем прогресс
            def _update_progress(idx=i):
                gen_label.configure(text=f"🖼 Генерация галереи... {idx + 1}/{len(display_moments)}")
            gallery.after(0, _update_progress)

        def _done():
            gen_label.configure(text=f"✅ Галерея готова ({len(display_moments)} моментов)")
        gallery.after(0, _done)

    threading.Thread(target=generate_thumbnails, daemon=True).start()


def _show_full_frame(parent, frame, moment):
    """Показывает полный кадр в отдельном окне."""
    win = ctk.CTkToplevel(parent)
    win.title(f"Кадр {moment['timestamp']}")

    annotated = _draw_boxes(frame, moment["objects"])
    # Ограничиваем размер до 1200px по ширине
    ctk_img = _frame_to_ctkimage(annotated, max_width=1200)

    label = ctk.CTkLabel(win, image=ctk_img, text="")
    label.image = ctk_img
    label.pack(padx=10, pady=10)

    obj_summary = ", ".join(
        f"{o['class']} {o['confidence']:.2f}" for o in moment["objects"]
    )
    ctk.CTkLabel(win, text=f"{moment['timestamp']} — {obj_summary}",
                 font=ctk.CTkFont(size=13)).pack(pady=5)


def launch():
    """Точка входа GUI."""
    from core import hardware, backend
    from core.detector import MilitaryDetector
    from core.classes import MILITARY_CLASSES
    from core import nvidia_client

    app = ctk.CTk()
    app.title("MuraveiVision — Military Object Detector")
    app.geometry("1200x800")

    detector = MilitaryDetector()

    # Заголовок
    ctk.CTkLabel(
        app,
        text="🎯 MuraveiVision",
        font=ctk.CTkFont(size=24, weight="bold"),
    ).pack(pady=15)

    # Информация о железе
    hw = hardware.detect_all()
    backend_str = backend.describe_backend(hw["gpu"], hw["cpu"])
    ctk.CTkLabel(
        app,
        text=f"⚙️ Backend: {backend_str}",
        font=ctk.CTkFont(size=14),
        text_color="#88ccff",
    ).pack(pady=5)

    # Фрейм управления
    ctrl = ctk.CTkFrame(app)
    ctrl.pack(padx=20, pady=10, fill="x")

    video_path_var = ctk.StringVar(value="")
    video_label = ctk.CTkLabel(ctrl, text="Видео не выбрано", text_color="gray")
    video_label.grid(row=0, column=1, padx=10, sticky="w")

    def choose_video():
        p = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv")],
        )
        if p:
            video_path_var.set(p)
            video_label.configure(text=os.path.basename(p), text_color="white")

    ctk.CTkButton(ctrl, text="📹 Выбрать видео", command=choose_video, width=180).grid(
        row=0, column=0, padx=10, pady=10
    )

    drone_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(ctrl, text="🚁 Режим дрона", variable=drone_var).grid(
        row=1, column=0, padx=10, pady=5, sticky="w"
    )

    # Чекбокс слоя 4 (облачная проверка)
    cloud_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        ctrl, text="☁️ Облачная проверка подозрительных (слой 4)", variable=cloud_var
    ).grid(row=2, column=0, padx=10, pady=5, sticky="w")

    conf_label = ctk.CTkLabel(ctrl, text="Чувствительность: 0.35")
    conf_label.grid(row=1, column=1, sticky="w")

    def update_conf(v):
        conf_label.configure(text=f"Чувствительность: {float(v):.2f}")

    conf_slider = ctk.CTkSlider(
        ctrl, from_=0.05, to=0.95, number_of_steps=18, command=update_conf
    )
    conf_slider.set(0.35)
    conf_slider.grid(row=1, column=2, padx=10, sticky="ew")

    # Прогресс и лог
    progress = ctk.CTkProgressBar(app)
    progress.pack(padx=20, fill="x")
    progress.set(0)

    log = ctk.CTkTextbox(app, height=400)
    log.pack(padx=20, pady=10, fill="both", expand=True)

    def log_msg(msg):
        log.insert("end", msg + "\n")
        log.see("end")

    log_msg("✅ Приложение готово к работе")
    log_msg(f"📋 Классов в словаре: {len(MILITARY_CLASSES)}")
    log_msg(f"🤖 Модель: {detector.model_name}")
    log_msg(f"🖥️ Провайдеры: {', '.join(detector.providers)}")

    # Проверка доступности слоя 4
    if nvidia_client.is_available():
        log_msg("☁️ Слой 4 (NVIDIA Vision) доступен")
    else:
        log_msg("☁️ Слой 4 недоступен офлайн — работаем на локальных слоях")

    start_btn = None

    # Переменные для повторного открытия галереи
    last_video_path = {"value": None}
    last_moments = {"value": []}

    def on_progress(percent, timestamp_str, objects, extra_log=None):
        """Колбэк прогресса из детектора (вызывается из рабочего потока).

        Аргументы:
          percent       — процент выполнения (0..100);
          timestamp_str — таймкод текущего кадра;
          objects       — список найденных объектов;
          extra_log     — опциональное сообщение для лога (напр. о стоп-кадрах).
        """
        def _update():
            progress.set(min(percent / 100.0, 1.0))
            if extra_log:
                log_msg(extra_log)
            elif objects:
                classes = [o["class"] for o in objects]
                log_msg(f"⏱️ {timestamp_str} | {len(objects)} объектов: {', '.join(classes[:5])}")
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
            progress.set(0)
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

            # Облачная проверка (слой 4), если включена
            cloud_annotations = {}
            if cloud_var.get() and results:
                cloud_annotations = _run_cloud_analysis(path, results)

            # Галерея (автооткрытие после анализа — существующая логика)
            if results:
                app.after(0, lambda: open_gallery(app, path, results, cloud_annotations))
            else:
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

    # Фрейм с кнопками "Начать анализ" и "Галерея"
    btn_frame = ctk.CTkFrame(app, fg_color="transparent")
    btn_frame.pack(pady=10)

    start_btn = ctk.CTkButton(
        btn_frame,
        text="🚀 Начать анализ",
        command=start_analysis,
        fg_color="#2ecc71",
        hover_color="#27ae60",
        height=45,
        width=220,
    )
    start_btn.pack(side="left", padx=10)

    def open_gallery_click():
        """Обработчик кнопки '🎞 Галерея'."""
        if not last_moments["value"]:
            messagebox.showinfo("Галерея", "Анализ ещё не выполнялся")
            return
        open_gallery(app, last_video_path["value"], last_moments["value"])

    gallery_btn = ctk.CTkButton(
        btn_frame,
        text="🎞 Галерея",
        command=open_gallery_click,
        fg_color="#3498db",
        hover_color="#2980b9",
        height=45,
        width=180,
    )
    gallery_btn.pack(side="left", padx=10)

    app.mainloop()


if __name__ == "__main__":
    launch()
