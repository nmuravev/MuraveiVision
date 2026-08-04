"""GUI приложения MuraveiVision (CustomTkinter)."""
import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def launch():
    """Точка входа GUI."""
    from core import hardware, backend
    from core.detector import MilitaryDetector

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
    log_msg(f"📋 Классов в словаре: {len(detector.classes)}")

    start_btn = None

    def run_analysis():
        path = video_path_var.get()
        if not path:
            messagebox.showwarning("Ошибка", "Сначала выберите видео!")
            start_btn.configure(state="normal", text="🚀 Начать анализ")
            return
        try:
            results = detector.analyze_video(
                path,
                drone_mode=drone_var.get(),
                confidence=conf_slider.get(),
                progress_callback=on_progress,
            )
            app.after(0, lambda: log_msg(f"✅ Найдено моментов: {len(results)}"))
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

    start_btn = ctk.CTkButton(
        app,
        text="🚀 Начать анализ",
        command=start_analysis,
        fg_color="#2ecc71",
        hover_color="#27ae60",
        height=45,
    )
    start_btn.pack(pady=10)

    app.mainloop()


if __name__ == "__main__":
    launch()