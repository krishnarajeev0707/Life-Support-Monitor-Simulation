import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import time

# COLOR PALETTE --------------------------------------------
C = {
    # Backgrounds
    "bg_main":        "#0a0a0a",
    "bg_panel":       "#111111",
    "bg_btn":         "#1d1d1d",
    "bg_plot":        "#111111",
    "bg_grid":        "#3a3a3a",

    # Text colors
    "text_header":    "#00ff88",
    "text_clock":     "#dddddd",
    "text_dim":       "#aaaaaa",
    "text_dimmer":    "#bbbbbb",
    "text_desc":      "#cccccc",
    "text_divider":   "#444455",

    # Vital value colors
    "vital_hr":       "#ff4444",
    "vital_spo2":     "#00aaff",
    "vital_bp":       "#ffaa00",
    "vital_rr":       "#44ffaa",
    "vital_temp":     "#cc88ff",

    # Wave colors
    "wave_ecg":       "#00ff88",
    "wave_spo2":      "#00aaff",
    "wave_resp":      "#ffaa00",

    # Wave title colors
    "wave_ok":        "#00aaff",
    "wave_crit":      "#ff4444",
    "resp_ok":        "#ffaa00",
    "resp_crit":      "#ff4444",

    # Condition colors
    "cond_normal":    "#00ff88",
    "cond_arrest":    "#ff3333",
    "cond_tachy":     "#ffaa00",
    "cond_brady":     "#ffdd00",
    "cond_hypoxia":   "#00aaff",
    "cond_htn":       "#ff6600",
    "cond_hypotherm": "#66ccff",
    "cond_septic":    "#cc44ff",
    "cond_vfib":      "#ff2266",
}


# ILLNESS PROFILES -----------------------------------------
PROFILES = {
    "Normal": {
        "color": C["cond_normal"], "hr": 75, "spo2": 98,
        "bp_sys": 120, "bp_dia": 80, "rr": 16, "temp": 36.8,
        "ecg_type": "normal", "desc": "All vitals within normal range.",
    },
    "Cardiac Arrest": {
        "color": C["cond_arrest"], "hr": 0, "spo2": 60,
        "bp_sys": 0, "bp_dia": 0, "rr": 0, "temp": 35.0,
        "ecg_type": "flatline", "desc": "No cardiac output. Immediate CPR required.",
    },
    "Tachycardia": {
        "color": C["cond_tachy"], "hr": 165, "spo2": 94,
        "bp_sys": 145, "bp_dia": 95, "rr": 24, "temp": 37.2,
        "ecg_type": "tachy", "desc": "Heart rate dangerously elevated (>150 bpm).",
    },
    "Bradycardia": {
        "color": C["cond_brady"], "hr": 35, "spo2": 91,
        "bp_sys": 88, "bp_dia": 55, "rr": 10, "temp": 36.1,
        "ecg_type": "brady", "desc": "Heart rate dangerously low (<40 bpm).",
    },
    "Hypoxia": {
        "color": C["cond_hypoxia"], "hr": 110, "spo2": 78,
        "bp_sys": 100, "bp_dia": 65, "rr": 28, "temp": 37.0,
        "ecg_type": "tachy", "desc": "Critically low blood oxygen saturation.",
    },
    "Hypertensive Crisis": {
        "color": C["cond_htn"], "hr": 95, "spo2": 95,
        "bp_sys": 210, "bp_dia": 130, "rr": 20, "temp": 37.5,
        "ecg_type": "normal", "desc": "Severely elevated blood pressure. Stroke risk.",
    },
    "Hypothermia": {
        "color": C["cond_hypotherm"], "hr": 42, "spo2": 89,
        "bp_sys": 85, "bp_dia": 50, "rr": 8, "temp": 32.1,
        "ecg_type": "brady", "desc": "Core body temperature critically low (<35°C).",
    },
    "Septic Shock": {
        "color": C["cond_septic"], "hr": 128, "spo2": 85,
        "bp_sys": 75, "bp_dia": 40, "rr": 30, "temp": 39.8,
        "ecg_type": "tachy", "desc": "Multi-organ failure. Systemic infection.",
    },
    "Ventricular Fibrillation": {
        "color": C["cond_vfib"], "hr": 0, "spo2": 55,
        "bp_sys": 0, "bp_dia": 0, "rr": 0, "temp": 35.5,
        "ecg_type": "vfib", "desc": "Chaotic ventricular activity. Defibrillation needed.",
    },
}

# ECG WAVEFORM GENERATORS ----------------------------------
def ecg_normal(t, hr=75):
    period = 60.0 / max(hr, 1)
    phase = (t % period) / period
    wave = np.zeros_like(phase)
    wave += 0.15 * np.exp(-((phase - 0.15) ** 2) / (2 * 0.003))
    wave -= 0.05 * np.exp(-((phase - 0.24) ** 2) / (2 * 0.001))
    wave += 1.0  * np.exp(-((phase - 0.27) ** 2) / (2 * 0.001))
    wave -= 0.15 * np.exp(-((phase - 0.30) ** 2) / (2 * 0.001))
    wave += 0.25 * np.exp(-((phase - 0.45) ** 2) / (2 * 0.005))
    return wave * 0.9
def ecg_flatline(t):
    return 0.015 * np.sin(2 * np.pi * 0.5 * t) + 0.008 * np.sin(2 * np.pi * 1.3 * t)
def ecg_vfib(t):
    w = (0.6 * np.sin(2 * np.pi * 7.3  * t) +
         0.5 * np.sin(2 * np.pi * 11.7 * t + 1.3) +
         0.4 * np.sin(2 * np.pi * 5.1  * t + 2.7) +
         0.3 * np.sin(2 * np.pi * 14.3 * t + 0.9) +
         0.2 * np.sin(2 * np.pi * 3.7  * t + 1.8) +
         0.2 * np.sin(2 * np.pi * 19.1 * t + 3.1))
    return w * 0.45
def ecg_tachy(t, hr=165):
    return ecg_normal(t, hr=hr)
def ecg_brady(t, hr=35):
    return ecg_normal(t, hr=hr)
def get_ecg(ecg_type, t, hr):
    if ecg_type == "normal":   return ecg_normal(t, hr)
    if ecg_type == "flatline": return ecg_flatline(t)
    if ecg_type == "vfib":     return ecg_vfib(t)
    if ecg_type == "tachy":    return ecg_tachy(t, hr)
    if ecg_type == "brady":    return ecg_brady(t, hr)
    return ecg_normal(t, hr)


# MAIN APP -------------------------------------------------
class LifeSupportMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Life Support Monitor — SciQuest 2026")
        self.root.configure(bg=C["bg_main"])
        self.root.geometry("1600x950")
        self.current_profile = "Normal"
        self.profile = PROFILES["Normal"]
        self._build_ui()
        self._start_animation()

    def _build_ui(self):
        top = tk.Frame(self.root, bg=C["bg_panel"], height=50)
        top.pack(fill="x", padx=0, pady=0)
        top.pack_propagate(False)
        tk.Label(top, text="🏥  LIFE SUPPORT MONITOR",
                font=("Courier New", 16, "bold"),
                bg=C["bg_panel"], fg=C["text_header"]).pack(side="left", padx=20, pady=10)
        self.time_label = tk.Label(top, text="", font=("Courier New", 13),
                                bg=C["bg_panel"], fg=C["text_clock"])
        self.time_label.pack(side="right", padx=20)
        self._update_time()

        main = tk.Frame(self.root, bg=C["bg_main"])
        main.pack(fill="both", expand=True, padx=10, pady=6)
        left = tk.Frame(main, bg=C["bg_main"])
        left.pack(side="left", fill="both", expand=True)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 5.5), facecolor=C["bg_main"])
        self.fig.subplots_adjust(hspace=0.45, left=0.08, right=0.97, top=0.93, bottom=0.08)
        self.wave_labels = ["ECG  (mV)", "SpO₂ Pleth", "Resp. Wave"]
        self.wave_colors = [C["wave_ecg"], C["wave_spo2"], C["wave_resp"]]
        self.lines = []

        for i, ax in enumerate(self.axes):
            ax.set_facecolor(C["bg_plot"])
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            ax.set_ylabel(self.wave_labels[i], color=C["text_dimmer"], fontsize=8, fontfamily="monospace")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")
            ax.set_xlim(0, 6)
            line, = ax.plot([], [], color=self.wave_colors[i], lw=1.4)
            self.lines.append(line)
            ax.yaxis.grid(True, color=C["bg_grid"], lw=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        right = tk.Frame(main, bg=C["bg_panel"], width=260)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        vitals_frame = tk.Frame(right, bg=C["bg_panel"])
        vitals_frame.pack(fill="x", padx=8, pady=(10, 4))
        tk.Label(vitals_frame, text="PATIENT VITALS",
                    font=("Courier New", 10, "bold"),
                    bg=C["bg_panel"], fg=C["text_dim"]).pack(anchor="w")
        self.vital_widgets = {}
        vitals_def = [
            ("HR",   "bpm",  C["vital_hr"]),
            ("SpO₂", "%",    C["vital_spo2"]),
            ("BP",   "mmHg", C["vital_bp"]),
            ("RR",   "/min", C["vital_rr"]),
            ("TEMP", "°C",   C["vital_temp"]),
        ]
        for key, unit, color in vitals_def:
            row = tk.Frame(vitals_frame, bg=C["bg_panel"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=key, font=("Courier New", 9),
                    bg=C["bg_panel"], fg=C["text_dimmer"], width=5, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, text="---", font=("Courier New", 22, "bold"),
                                bg=C["bg_panel"], fg=color, anchor="w")
            val_lbl.pack(side="left")
            tk.Label(row, text=unit, font=("Courier New", 9),
                        bg=C["bg_panel"], fg=C["text_dim"]).pack(side="left", padx=(2, 0))
            self.vital_widgets[key] = val_lbl

        self.desc_label = tk.Label(right, text="", font=("Courier New", 9),
                                    bg=C["bg_panel"], fg=C["text_desc"],
                                    wraplength=240, justify="left")
        self.desc_label.pack(padx=10, pady=(6, 0), anchor="w")
        tk.Frame(right, bg=C["text_divider"], height=1).pack(fill="x", padx=8, pady=10)
        tk.Label(right, text="SELECT CONDITION", font=("Courier New", 10, "bold"),
                    bg=C["bg_panel"], fg=C["text_dim"]).pack(padx=10, anchor="w")

        btn_frame = tk.Frame(right, bg=C["bg_panel"])
        btn_frame.pack(fill="both", expand=True, padx=8, pady=4)
        self.buttons = {}
        for name, profile in PROFILES.items():
            btn = tk.Button(
                btn_frame, text=name, font=("Courier New", 9, "bold"),
                bg=C["bg_btn"], fg=profile["color"],
                activebackground=profile["color"], activeforeground=C["bg_main"],
                relief="flat", bd=0, cursor="hand2",
                command=lambda n=name: self._switch_profile(n), pady=5
            )
            btn.pack(fill="x", pady=2)
            self.buttons[name] = btn
        self._highlight_button("Normal")

    def _update_time(self):
        from datetime import datetime
        self.time_label.config(text=datetime.now().strftime("  %H:%M:%S   %d %b %Y"))
        self.root.after(1000, self._update_time)

    def _switch_profile(self, name):
        self.current_profile = name
        self.profile = PROFILES[name]
        self.start_time = time.time()
        self._highlight_button(name)
        self._update_vitals()

    def _highlight_button(self, active):
        for name, btn in self.buttons.items():
            if name == active:
                btn.config(bg=PROFILES[name]["color"], fg=C["bg_main"])
            else:
                btn.config(bg=C["bg_btn"], fg=PROFILES[name]["color"])

    def _update_vitals(self):
        p = self.profile
        self.vital_widgets["HR"].config(text=str(p["hr"]) if p["hr"] > 0 else "---")
        self.vital_widgets["SpO₂"].config(text=str(p["spo2"]))
        self.vital_widgets["BP"].config(
            text=f"{p['bp_sys']}/{p['bp_dia']}" if p["bp_sys"] > 0 else "---/---")
        self.vital_widgets["RR"].config(text=str(p["rr"]) if p["rr"] > 0 else "---")
        self.vital_widgets["TEMP"].config(text=str(p["temp"]))
        self.desc_label.config(text=p["desc"])

    def _start_animation(self):
        self._update_vitals()
        self.start_time = time.time()
        self._animate_loop()
    def _animate_loop(self):
        self._animate()
        self.root.after(40, self._animate_loop)
    def _animate(self):
        elapsed = time.time() - self.start_time
        p = self.profile
        color = p["color"]
        t_end = elapsed
        t_start = t_end - 6
        t = np.linspace(t_start, t_end, 600)

        # ECG
        ecg = get_ecg(p["ecg_type"], t, p["hr"])
        self.lines[0].set_data(t, ecg)
        self.lines[0].set_color(color)
        self.axes[0].set_xlim(t_start, t_end)
        self.axes[0].set_ylim(-0.4, 1.4)
        self.axes[0].set_title(
            f"ECG   {'♥ ' + str(p['hr']) + ' BPM' if p['hr'] > 0 else '● FLATLINE'}",
            color=color, fontsize=9, fontfamily="monospace", loc="left", pad=3)

        # SpO2
        freq = p["hr"] / 60.0 if p["hr"] > 0 else 0
        if p["spo2"] < 70:
            spo2_wave = 0.08 * np.sin(2 * np.pi * 0.3 * t) + 0.04 * np.sin(2 * np.pi * 0.7 * t)
        else:
            spo2_wave = (0.5 * np.sin(2 * np.pi * freq * t) +
                         0.2 * np.sin(2 * np.pi * freq * 2 * t))
        self.lines[1].set_data(t, spo2_wave)
        self.lines[1].set_color(C["wave_spo2"])
        self.axes[1].set_xlim(t_start, t_end)
        self.axes[1].set_ylim(-1.0, 1.2)
        spo2_color = C["wave_ok"] if p["spo2"] >= 90 else C["wave_crit"]
        self.axes[1].set_title(
            f"SpO₂  {p['spo2']}%",
            color=spo2_color, fontsize=9, fontfamily="monospace", loc="left", pad=3)

        # Respiration
        if p["rr"] == 0:
            resp_wave = 0.02 * np.sin(2 * np.pi * 0.2 * t)
        else:
            resp_wave = 0.6 * np.sin(2 * np.pi * (p["rr"] / 60.0) * t)
        self.lines[2].set_data(t, resp_wave)
        self.lines[2].set_color(C["wave_resp"])
        self.axes[2].set_xlim(t_start, t_end)
        self.axes[2].set_ylim(-1.0, 1.0)
        rr_color = C["resp_ok"] if p["rr"] >= 8 else C["resp_crit"]
        self.axes[2].set_title(
            f"RESP  {p['rr']}/min",
            color=rr_color, fontsize=9, fontfamily="monospace", loc="left", pad=3)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = LifeSupportMonitor(root)
    root.mainloop()