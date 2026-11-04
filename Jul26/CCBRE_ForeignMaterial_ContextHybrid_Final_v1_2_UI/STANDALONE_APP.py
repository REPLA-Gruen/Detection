#!/usr/bin/env python3
"""CCBRE AI-LAB unified Stage-0 + A/B/C visual application.

The main window deliberately contains no permanent model/threshold control rows.
Advanced controls live in menus/dialogs so image and analysis results receive
nearly the entire screen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import yaml
from PIL import Image, ImageTk

from context_inference import ContextClassifier
from phase4_inference import Pipeline, norm
from phase4_rendering import (
    FO_CLASS_COLORS,
    PRED_ERROR_COLOR,
    PRED_UNCERTAIN_COLOR,
    render_unlabeled,
    render_with_gt,
)
from combined_rendering import prepend_context_banner
from evaluate_TCoreOther_comprehensive import parse_ground_truth, same_class_match

ROOT = Path(__file__).resolve().parent

ANALYSIS_ZOOM_PRESETS = (80, 100, 120, 150, 200, 500)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def fmt(v, T):
    return Path(str(v).replace("{T}", str(T))).expanduser().resolve()


def bundled(rel):
    return (
        Path(getattr(sys, "_MEIPASS", ROOT)) / rel
        if hasattr(sys, "_MEIPASS")
        else ROOT / rel
    )


def default_config():
    if getattr(sys, "frozen", False):
        p = Path(sys.executable).resolve().parent / "config.json"
        if p.is_file():
            return p
    return ROOT / "config.json"


def category_counts(gt_count, verified, same_tp, exact_count):
    uncertain = max(0, verified - exact_count)
    fp = max(0, exact_count - same_tp) + uncertain
    fn = max(0, gt_count - same_tp)
    if gt_count == 0 and verified == 0:
        return "correct_negative", fp, fn, uncertain
    if fp == 0 and fn == 0:
        return "perfect_detection", fp, fn, uncertain
    if fp > 0 and fn == 0:
        return "false_positive_only", fp, fn, uncertain
    if fn > 0 and fp == 0:
        return "false_negative_only", fp, fn, uncertain
    return "mixed_error", fp, fn, uncertain


def find_yaml(start):
    cur = start.resolve()
    for _ in range(6):
        for n in ("dataset.yaml", "data.yaml", "dataset.yml", "data.yml"):
            p = cur / n
            if p.is_file():
                return p
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def names_from_yaml(p):
    if not p:
        return None
    raw = yaml.safe_load(p.read_text(encoding="utf-8-sig")) or {}
    n = raw.get("names")
    if isinstance(n, list):
        return {i: str(v) for i, v in enumerate(n)}
    if isinstance(n, dict):
        return {int(k): str(v) for k, v in n.items()}
    return None


def sibling_labels(image):
    out = [image.with_suffix(".txt")]
    parts = list(image.parts)
    for i, part in enumerate(parts):
        if part.casefold() == "images":
            q = parts.copy()
            q[i] = "labels"
            out.append(Path(*q).with_suffix(".txt"))
    return list(dict.fromkeys(out))


def context_gt_from_folder(image):
    parts = [p.casefold() for p in image.parts]
    for cls in ("container", "wagon"):
        if cls in parts:
            return cls
    return None


def context_gt_from_suffix(image):
    s = image.stem.casefold()
    if s.endswith("w"):
        return "wagon"
    if s.endswith(("c", "b", "t")):
        return "container"
    return None


def rgba_to_hex(color):
    r, g, b = color[:3]
    return f"#{r:02x}{g:02x}{b:02x}"


class App:
    def __init__(self, root, config_path):
        self.root = root
        self.config_path = Path(config_path).resolve()
        self.c = load(self.config_path)
        self.T = int(self.c["threshold"])
        self.runs = fmt(self.c["runs_root"], self.T)
        self.final = self.runs / "FINAL_MODELS"
        self.manifest = load(self.final / "final_model_manifest.json")

        self.context_path = Path(self.manifest["models"]["context"])
        self.a_path = Path(self.manifest["models"]["stage_a"])
        self.b_path = Path(self.manifest["models"]["stage_b"])
        self.c_path = Path(self.manifest["models"]["stage_c"])
        self.op = tk.StringVar(value=self.c["default_operating_point"])

        app_cfg = self.c.get("app", {})
        self.show_analysis = tk.BooleanVar(
            value=bool(app_cfg.get("show_analysis_panel_by_default", True))
        )
        self.show_full_render = tk.BooleanVar(
            value=bool(app_cfg.get("show_saved_render_panel_in_viewer_by_default", False))
        )
        self.context_suffix_gt = tk.BooleanVar(
            value=bool(app_cfg.get("context_gt_from_filename_default", False))
        )
        self.analysis_font_size = int(app_cfg.get("analysis_font_size", 10))
        self.analysis_zoom_percent = int(app_cfg.get("analysis_zoom_percent", 100))
        self.window_width = int(app_cfg.get("window_width", 1580))
        self.window_height = int(app_cfg.get("window_height", 980))
        self.image_pane_percent = int(app_cfg.get("image_pane_percent", 67))
        self.min_window_width = int(app_cfg.get("min_window_width", 1000))
        self.min_window_height = int(app_cfg.get("min_window_height", 700))

        self.fullscreen = False
        self.context_model = None
        self.pipe = None
        self.pipeline_key = None
        self.image_path = None
        self.original = None
        self.display_image = None
        self.full_render = None
        self.image_only = None
        self.photo = None
        self.result = None
        self.busy = False

        root.title("CCBRE AI-LAB — Context + Foreign Material Analysis")
        root.geometry(f"{self.window_width}x{self.window_height}")
        root.minsize(self.min_window_width, self.min_window_height)

        self.make_menu()
        self.make_header()
        self.make_body()
        self.make_footer()
        self.bind_keys()
        self.root.after(120, self.apply_pane_ratio)
        self.status.set("Ready — select an image (Ctrl+O).")

    def make_menu(self):
        m = tk.Menu(self.root)
        self.root.config(menu=m)

        f = tk.Menu(m, tearoff=0)
        m.add_cascade(label="File", menu=f)
        f.add_command(
            label="Open image and analyze…",
            accelerator="Ctrl+O",
            command=self.open_image,
        )
        f.add_command(
            label="Re-analyze current image",
            accelerator="Ctrl+R",
            command=self.start,
        )
        f.add_command(
            label="Save rendered result as…",
            accelerator="Ctrl+S",
            command=self.save_as,
        )
        f.add_command(label="Open output folder", command=self.open_output)
        f.add_separator()
        f.add_command(label="Exit", command=self.root.destroy)

        models = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Models", menu=models)
        models.add_command(
            label="Use FINAL_MODELS (recommended)", command=self.reset_models
        )
        models.add_command(
            label="Use optional D1 YOLOv8m context reference",
            command=self.use_context_reference,
        )
        custom = tk.Menu(models, tearoff=0)
        models.add_cascade(label="Custom checkpoints", menu=custom)
        custom.add_command(
            label="Stage 0 context classifier…",
            command=lambda: self.browse_model("context"),
        )
        custom.add_command(
            label="Stage A Any-FM detector…",
            command=lambda: self.browse_model("a"),
        )
        custom.add_command(
            label="Stage B binary verifier…",
            command=lambda: self.browse_model("b"),
        )
        custom.add_command(
            label="Stage C exact classifier…",
            command=lambda: self.browse_model("c"),
        )
        models.add_separator()
        models.add_command(label="Show selected model paths", command=self.show_models)
        models.add_command(label="Validate selected models", command=self.validate_models)

        opt = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Options", menu=opt)
        ops = tk.Menu(opt, tearoff=0)
        opt.add_cascade(label="Operating point", menu=ops)
        for name in self.c["operating_points"]:
            ops.add_radiobutton(
                label=name, variable=self.op, value=name, command=self.invalidate
            )
        opt.add_checkbutton(
            label="Use filename suffix as context GT (project audit only)",
            variable=self.context_suffix_gt,
        )

        view = tk.Menu(m, tearoff=0)
        m.add_cascade(label="View", menu=view)
        view.add_checkbutton(
            label="Show analysis panel",
            accelerator="F9",
            variable=self.show_analysis,
            command=self.toggle_analysis,
        )
        view.add_checkbutton(
            label="Show saved-render information panel in viewer",
            variable=self.show_full_render,
            command=self.refresh_image,
        )
        view.add_separator()
        view.add_command(
            label="Display settings…", command=self.show_display_settings
        )
        zoom = tk.Menu(view, tearoff=0)
        view.add_cascade(label="Analysis text zoom", menu=zoom)
        self.analysis_zoom_var = tk.IntVar(value=self.analysis_zoom_percent)
        for percent in ANALYSIS_ZOOM_PRESETS:
            zoom.add_radiobutton(
                label=f"{percent}%",
                variable=self.analysis_zoom_var,
                value=percent,
                command=lambda p=percent: self.set_analysis_zoom(p),
            )
        zoom.add_separator()
        zoom.add_command(label="Custom…", command=self.show_custom_analysis_zoom)
        view.add_command(
            label="Zoom in analysis text",
            accelerator="Ctrl++",
            command=lambda: self.bump_analysis_zoom(1),
        )
        view.add_command(
            label="Zoom out analysis text",
            accelerator="Ctrl+-",
            command=lambda: self.bump_analysis_zoom(-1),
        )
        view.add_command(
            label="Reset analysis text to 100%",
            accelerator="Ctrl+0",
            command=lambda: self.set_analysis_zoom(100),
        )
        layout = tk.Menu(view, tearoff=0)
        view.add_cascade(label="Pane layout", menu=layout)
        layout.add_command(
            label="Image 70% / Analysis 30%", command=lambda: self.set_pane_percent(70)
        )
        layout.add_command(
            label="Image 60% / Analysis 40%", command=lambda: self.set_pane_percent(60)
        )
        layout.add_command(
            label="Image 50% / Analysis 50%", command=lambda: self.set_pane_percent(50)
        )
        view.add_command(label="Reset display defaults", command=self.reset_layout)
        view.add_separator()
        view.add_command(
            label="Fullscreen", accelerator="F11", command=self.toggle_fullscreen
        )

        tools = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Tools", menu=tools)
        tools.add_command(label="Validate selected models", command=self.validate_models)
        tools.add_command(label="Open output folder", command=self.open_output)

        h = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Help", menu=h)
        h.add_command(
            label="User guide & color legend…", command=self.show_help_window
        )
        h.add_command(label="Keyboard shortcuts", command=self.show_shortcuts)
        h.add_separator()
        h.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About",
                "CCBRE AI-LAB — ContextHybrid Final UI v1.2\n\n"
                "Stage 0: Container/Wagon context\n"
                "Stages A/B/C: foreign-material detection, verification, and exact classification",
            ),
        )

    def make_header(self):
        h = ttk.Frame(self.root, padding=(8, 4))
        h.pack(fill="x")
        logo = bundled(Path("assets") / "CCBRE_AI_LAB_Logo.png")
        if logo.is_file():
            im = Image.open(logo).convert("RGBA")
            im.thumbnail((62, 48), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(im)
            ttk.Label(h, image=self.logo).pack(side="left", padx=(0, 8))
        text = ttk.Frame(h)
        text.pack(side="left", fill="x", expand=True)
        ttk.Label(
            text,
            text="AI-LAB CCBRE.EU — Context + Foreign Material Analysis",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        self.headline = tk.StringVar(
            value="Stage 0: context | Stages A/B/C: foreign material"
        )
        ttk.Label(text, textvariable=self.headline, font=("Segoe UI", 9)).pack(
            anchor="w"
        )

    def make_body(self):
        self.pane = ttk.Panedwindow(self.root, orient="horizontal")
        self.pane.pack(fill="both", expand=True, padx=4, pady=2)
        self.image_frame = ttk.Frame(self.pane)
        self.analysis_frame = ttk.Frame(self.pane)
        self.pane.add(self.image_frame, weight=4)
        self.pane.add(self.analysis_frame, weight=2)

        self.canvas = tk.Canvas(
            self.image_frame, bg="#171717", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.refresh_image())

        self.text = tk.Text(
            self.analysis_frame,
            wrap="word",
            font=("Consolas", self.effective_analysis_font_size()),
            borderwidth=0,
        )
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        if not self.show_analysis.get():
            self.pane.forget(self.analysis_frame)

    def make_footer(self):
        self.status = tk.StringVar()
        ttk.Label(
            self.root, textvariable=self.status, anchor="w", padding=(6, 3)
        ).pack(fill="x")

    def bind_keys(self):
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-r>", lambda e: self.start())
        self.root.bind("<Control-s>", lambda e: self.save_as())
        self.root.bind(
            "<F9>",
            lambda e: (
                self.show_analysis.set(not self.show_analysis.get()),
                self.toggle_analysis(),
            ),
        )
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Control-plus>", lambda e: self.bump_analysis_zoom(1))
        self.root.bind("<Control-equal>", lambda e: self.bump_analysis_zoom(1))
        self.root.bind("<Control-minus>", lambda e: self.bump_analysis_zoom(-1))
        self.root.bind("<Control-0>", lambda e: self.set_analysis_zoom(100))

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def toggle_analysis(self):
        shown = str(self.analysis_frame) in self.pane.panes()
        if self.show_analysis.get() and not shown:
            self.pane.add(self.analysis_frame, weight=2)
            self.root.after(80, self.apply_pane_ratio)
        elif not self.show_analysis.get() and shown:
            self.pane.forget(self.analysis_frame)

    def set_pane_percent(self, percent):
        self.image_pane_percent = max(35, min(85, int(percent)))
        self.apply_pane_ratio()
        self.status.set(
            f"Pane layout: image {self.image_pane_percent}% / analysis {100-self.image_pane_percent}%."
        )

    def apply_pane_ratio(self):
        if not self.show_analysis.get() or len(self.pane.panes()) < 2:
            return
        self.root.update_idletasks()
        total = max(200, self.pane.winfo_width())
        pos = int(total * self.image_pane_percent / 100.0)
        pos = max(150, min(total - 150, pos))
        try:
            self.pane.sashpos(0, pos)
        except tk.TclError:
            pass

    def effective_analysis_font_size(self):
        return max(6, int(round(self.analysis_font_size * self.analysis_zoom_percent / 100.0)))

    def apply_analysis_zoom(self, announce=True):
        size = self.effective_analysis_font_size()
        self.text.configure(font=("Consolas", size))
        if hasattr(self, "analysis_zoom_var"):
            self.analysis_zoom_var.set(self.analysis_zoom_percent)
        if announce:
            self.status.set(
                f"Analysis text zoom: {self.analysis_zoom_percent}% "
                f"(effective font {size} pt)."
            )

    def set_analysis_zoom(self, percent):
        self.analysis_zoom_percent = max(50, min(500, int(percent)))
        self.apply_analysis_zoom(True)

    def show_custom_analysis_zoom(self):
        percent = simpledialog.askinteger(
            "Analysis text zoom",
            "Enter analysis text zoom (50–500%):",
            parent=self.root,
            initialvalue=self.analysis_zoom_percent,
            minvalue=50,
            maxvalue=500,
        )
        if percent is not None:
            self.set_analysis_zoom(percent)

    def bump_analysis_zoom(self, direction):
        presets = list(ANALYSIS_ZOOM_PRESETS)
        current = self.analysis_zoom_percent
        if direction > 0:
            candidates = [p for p in presets if p > current]
            target = candidates[0] if candidates else min(500, current + 25)
        else:
            candidates = [p for p in presets if p < current]
            target = candidates[-1] if candidates else max(50, current - 10)
        self.set_analysis_zoom(target)

    def reset_layout(self):
        app_cfg = self.c.get("app", {})
        self.window_width = int(app_cfg.get("window_width", 1580))
        self.window_height = int(app_cfg.get("window_height", 980))
        self.image_pane_percent = int(app_cfg.get("image_pane_percent", 67))
        self.analysis_font_size = int(app_cfg.get("analysis_font_size", 10))
        self.analysis_zoom_percent = int(app_cfg.get("analysis_zoom_percent", 100))
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.apply_analysis_zoom(False)
        self.root.after(100, self.apply_pane_ratio)
        self.status.set("Display reset to configured defaults.")

    def show_display_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Display settings")
        win.transient(self.root)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=14)
        frame.pack(fill="both", expand=True)

        width_var = tk.IntVar(value=self.root.winfo_width() or self.window_width)
        height_var = tk.IntVar(value=self.root.winfo_height() or self.window_height)
        pane_var = tk.IntVar(value=self.image_pane_percent)
        font_var = tk.IntVar(value=self.analysis_font_size)
        zoom_var = tk.IntVar(value=self.analysis_zoom_percent)

        rows = [
            ("Window width (px)", width_var, 900, 3840, 20),
            ("Window height (px)", height_var, 650, 2160, 20),
            ("Image pane width (%)", pane_var, 35, 85, 1),
            ("Analysis base font size (pt)", font_var, 8, 28, 1),
            ("Analysis text zoom (%)", zoom_var, 50, 500, 10),
        ]
        for r, (label, var, lo, hi, inc) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w", pady=5)
            ttk.Spinbox(
                frame, from_=lo, to=hi, increment=inc, textvariable=var, width=10
            ).grid(row=r, column=1, sticky="e", padx=(18, 0), pady=5)

        note = (
            "Tip: the vertical divider between the image and analysis panes can also be dragged directly.\n"
            "Window height changes the height of both side-by-side panes.\n"
            "Analysis text zoom can also be selected directly as 80/100/120/150/200/500% under View."
        )
        ttk.Label(frame, text=note, wraplength=420, justify="left").grid(
            row=len(rows), column=0, columnspan=2, sticky="w", pady=(10, 12)
        )

        def apply(save=False):
            try:
                w = max(900, min(3840, int(width_var.get())))
                h = max(650, min(2160, int(height_var.get())))
                pane = max(35, min(85, int(pane_var.get())))
                font = max(8, min(28, int(font_var.get())))
                zoom_percent = max(50, min(500, int(zoom_var.get())))
            except (TypeError, ValueError, tk.TclError):
                messagebox.showerror("Display settings", "Please enter valid numeric values.", parent=win)
                return
            self.window_width = w
            self.window_height = h
            self.image_pane_percent = pane
            self.analysis_font_size = font
            self.analysis_zoom_percent = zoom_percent
            self.root.geometry(f"{w}x{h}")
            self.apply_analysis_zoom(False)
            self.root.after(100, self.apply_pane_ratio)
            if save:
                app = self.c.setdefault("app", {})
                app["window_width"] = w
                app["window_height"] = h
                app["image_pane_percent"] = pane
                app["analysis_font_size"] = font
                app["analysis_zoom_percent"] = zoom_percent
                try:
                    self.config_path.write_text(
                        json.dumps(self.c, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    self.status.set("Display settings applied and saved as defaults.")
                except OSError as exc:
                    messagebox.showwarning(
                        "Display settings",
                        f"Settings were applied for this session, but config.json could not be updated:\n\n{exc}",
                        parent=win,
                    )
            else:
                self.status.set("Display settings applied for this session.")

        buttons = ttk.Frame(frame)
        buttons.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Apply", command=lambda: apply(False)).pack(
            side="left", padx=4
        )
        ttk.Button(buttons, text="Save as default", command=lambda: apply(True)).pack(
            side="left", padx=4
        )
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="left", padx=4)

    def show_help_window(self):
        win = tk.Toplevel(self.root)
        win.title("CCBRE AI-LAB Help")
        win.geometry("760x610")
        win.minsize(620, 480)
        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        quick = ttk.Frame(notebook, padding=14)
        colors = ttk.Frame(notebook, padding=14)
        controls = ttk.Frame(notebook, padding=14)
        notebook.add(quick, text="Quick start")
        notebook.add(colors, text="Color legend")
        notebook.add(controls, text="Controls & metrics")

        quick_text = (
            "1. File > Open image and analyze… (Ctrl+O). Analysis starts automatically.\n\n"
            "2. Stage 0 estimates the acquisition context (Container/Wagon). It is metadata and does not gate the foreign-material pipeline.\n\n"
            "3. Stage A proposes foreign-material boxes, Stage B verifies FO versus background, and Stage C assigns BAW-Sack, Papier, Plastiksack, or Other.\n\n"
            "4. The photo itself uses compact IDs only (P1, P2, …). Prediction-box color identifies the FO type using the Color legend. A verified object whose Stage-C confidence is insufficient is retained in purple as class uncertain.\n\n"
            "5. The annotated JPG and JSON are saved automatically under APP_OUTPUTS. If a matching YOLO label exists, GT comparison is performed automatically."
        )
        ttk.Label(quick, text=quick_text, wraplength=690, justify="left").pack(
            anchor="nw"
        )

        ttk.Label(
            colors,
            text="Foreign-material prediction colors",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        legend = [
            ("BAW-Sack", FO_CLASS_COLORS["baw-sack"]),
            ("Papier", FO_CLASS_COLORS["papier"]),
            ("Plastiksack", FO_CLASS_COLORS["plastiksack"]),
            ("Other", FO_CLASS_COLORS["other"]),
            ("Class uncertain", PRED_UNCERTAIN_COLOR),
        ]
        for name, color in legend:
            row = ttk.Frame(colors)
            row.pack(fill="x", pady=4)
            tk.Label(row, width=4, height=1, bg=rgba_to_hex(color), relief="solid", bd=1).pack(
                side="left", padx=(0, 10)
            )
            ttk.Label(row, text=name).pack(side="left")

        ttk.Separator(colors).pack(fill="x", pady=12)
        ttk.Label(
            colors,
            text="GT comparison overlays",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            colors,
            text=(
                "Ground-truth boxes keep the audit colors: green = matched and orange = missed. "
                "Prediction boxes keep their material-class color. A wrong/extra exact-class prediction receives an additional red inner outline."
            ),
            wraplength=690,
            justify="left",
        ).pack(anchor="w")
        redrow = ttk.Frame(colors)
        redrow.pack(fill="x", pady=8)
        tk.Label(redrow, width=4, height=1, bg=rgba_to_hex(PRED_ERROR_COLOR), relief="solid", bd=1).pack(
            side="left", padx=(0, 10)
        )
        ttk.Label(redrow, text="Red inner outline = wrong / extra prediction in GT comparison").pack(side="left")

        controls_text = (
            "Layout\n"
            "• Drag the divider between Image and Analysis Results at any time.\n"
            "• View > Display settings… changes window width, window height, pane ratio, base font size, and analysis text zoom.\n"
            "• View > Analysis text zoom provides 80%, 100%, 120%, 150%, 200%, and 500% presets plus a Custom… value from 50–500%.\n"
            "• View > Pane layout provides quick 70/30, 60/40, and 50/50 presets.\n\n"
            "Confidence labels\n"
            "• Det = Stage-A detector confidence.\n"
            "• Ver = Stage-B ForeignMaterial probability.\n"
            "• Cls = Stage-C exact-class confidence.\n\n"
            "Keyboard\n"
            "• Ctrl+O open/analyze image\n"
            "• Ctrl+R re-analyze current image\n"
            "• Ctrl+S save rendered result\n"
            "• Ctrl++ / Ctrl+- step through analysis text zoom levels\n"
            "• Ctrl+0 reset analysis text zoom to 100%\n"
            "• F9 show/hide analysis panel\n"
            "• F11 fullscreen\n\n"
            "Viewer\n"
            "The default viewer shows the annotated photo only. Prediction labels on the photo are compact IDs (P1, P2, …); use box color plus the Help legend or Analysis Results to identify the FO type. The saved diagnostic information panel can be shown using View > Show saved-render information panel in viewer."
        )
        ttk.Label(controls, text=controls_text, wraplength=690, justify="left").pack(
            anchor="nw"
        )

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 10))

    def show_shortcuts(self):
        messagebox.showinfo(
            "Keyboard shortcuts",
            "Ctrl+O   Open image and analyze\n"
            "Ctrl+R   Re-analyze current image\n"
            "Ctrl+S   Save rendered result\n"
            "Ctrl++   Increase analysis text zoom\n"
            "Ctrl+-   Decrease analysis text zoom\n"
            "Ctrl+0   Reset analysis text zoom to 100%\n"
            "F9       Show/hide analysis panel\n"
            "F11      Fullscreen",
        )

    def reset_models(self):
        self.context_path = Path(self.manifest["models"]["context"])
        self.a_path = Path(self.manifest["models"]["stage_a"])
        self.b_path = Path(self.manifest["models"]["stage_b"])
        self.c_path = Path(self.manifest["models"]["stage_c"])
        self.invalidate()
        self.status.set("Reset to FINAL_MODELS.")

    def use_context_reference(self):
        ref = self.manifest.get("optional_context_reference_yolov8m")
        if not ref or not Path(ref).is_file():
            messagebox.showwarning(
                "D1 reference unavailable",
                "No optional YOLOv8m context reference is present in FINAL_MODELS.",
            )
            return
        self.context_path = Path(ref)
        self.invalidate()
        self.status.set("Using optional D1 YOLOv8m context reference.")

    def browse_model(self, key):
        p = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if not p:
            return
        if key == "context":
            self.context_path = Path(p)
        elif key == "a":
            self.a_path = Path(p)
        elif key == "b":
            self.b_path = Path(p)
        else:
            self.c_path = Path(p)
        self.invalidate()
        self.status.set(f"Custom {key} model selected.")

    def show_models(self):
        messagebox.showinfo(
            "Selected models",
            f"Stage 0:\n{self.context_path}\n\n"
            f"Stage A:\n{self.a_path}\n\n"
            f"Stage B:\n{self.b_path}\n\n"
            f"Stage C:\n{self.c_path}\n\n"
            f"Operating point: {self.op.get()}",
        )

    def validate_models(self, quiet=False):
        missing = [
            str(p)
            for p in (self.context_path, self.a_path, self.b_path, self.c_path)
            if not Path(p).is_file()
        ]
        if missing:
            if not quiet:
                messagebox.showerror("Missing models", "\n".join(missing))
            raise FileNotFoundError("\n".join(missing))
        if not quiet:
            messagebox.showinfo("Models valid", "All four selected checkpoints exist.")

    def invalidate(self):
        self.context_model = None
        self.pipe = None
        self.pipeline_key = None

    def load_models(self):
        self.validate_models(True)
        key = (
            str(self.context_path),
            str(self.a_path),
            str(self.b_path),
            str(self.c_path),
            self.op.get(),
        )
        if self.pipeline_key == key:
            return
        self.status.set("Loading selected models…")
        self.root.update_idletasks()
        self.context_model = ContextClassifier(
            self.context_path,
            self.c["device"],
            self.c["context"]["app"]["imgsz"],
        )
        self.pipe = Pipeline(
            self.a_path,
            self.b_path,
            self.c_path,
            self.c["operating_points"][self.op.get()],
            self.c["device"],
        )
        self.pipeline_key = key

    def open_image(self):
        p = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not p:
            return
        self.image_path = Path(p)
        self.original = Image.open(self.image_path).convert("RGB")
        self.display_image = self.original
        self.full_render = None
        self.image_only = None
        self.result = None
        self.headline.set(self.image_path.name)
        self.status.set("Image selected — analysis starts automatically.")
        self.set_text(
            "Image selected.\n\nAnalysis is starting automatically. Advanced settings are under the menus."
        )
        self.refresh_image()
        self.start()

    def start(self):
        if not self.image_path:
            return
        if self.busy:
            return
        self.busy = True
        self.status.set("Analyzing Stage 0 + A/B/C…")
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        try:
            self.load_models()
            ctx = self.context_model.analyze(self.image_path)
            image, fm = self.pipe.analyze(self.image_path)
            gt = None
            names = None
            label = None
            for cand in sibling_labels(self.image_path):
                if cand.is_file():
                    label = cand
                    names = names_from_yaml(find_yaml(cand.parent)) or {
                        0: "BAW-Sack",
                        1: "Papier",
                        2: "Plastiksack",
                        3: "Other",
                    }
                    gt = parse_ground_truth(cand, image.size, names)
                    break

            matches = []
            exact = []
            category = None
            fp = fn = uncertain = 0
            if gt is not None:
                target_id = {norm(n): i for i, n in names.items()}
                for d in fm["detections"]:
                    if d["class_name"] is None:
                        continue
                    mapped = target_id.get(norm(d["class_name"]))
                    if mapped is not None:
                        exact.append(
                            {
                                "class_id": int(mapped),
                                "bbox": d["bbox"],
                                "confidence": d["detector_confidence"]
                                * d["verifier_fo_probability"]
                                * d["exact_confidence"],
                            }
                        )
                matches, _, _ = same_class_match(gt, exact)
                category, fp, fn, uncertain = category_counts(
                    len(gt), len(fm["detections"]), len(matches), len(exact)
                )
                render = render_with_gt(
                    image, fm, gt, exact, matches, names, category, fp, fn
                )
            else:
                render = render_unlabeled(image, fm)
                uncertain = sum(d["class_name"] is None for d in fm["detections"])

            context_gt = context_gt_from_folder(self.image_path)
            if context_gt is None and self.context_suffix_gt.get():
                context_gt = context_gt_from_suffix(self.image_path)

            full = prepend_context_banner(render, ctx, context_gt)
            image_only = full.crop(
                (0, full.height - image.height, full.width, full.height)
            )

            outroot = fmt(self.c["app"]["output_root"], self.T)
            sub = "GT_COMPARISON/" + category if category else "UNLABELED_INFERENCE"
            folder = outroot / sub
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            jpg = folder / f"{self.image_path.stem}__context_hybrid_{stamp}.jpg"
            js = jpg.with_suffix(".json")
            full.save(jpg, quality=95)

            record = {
                "image": str(self.image_path),
                "context": ctx,
                "context_ground_truth": context_gt,
                "foreign_material": fm,
                "foreign_gt_label": str(label) if label else None,
                "diagnostic_category": category,
                "tp": len(matches),
                "fp": fp,
                "fn": fn,
                "class_uncertain_verified_objects": uncertain,
                "models_used": {
                    "context": str(self.context_path),
                    "stage_a": str(self.a_path),
                    "stage_b": str(self.b_path),
                    "stage_c": str(self.c_path),
                    "operating_point": self.op.get(),
                },
                "rendered_image": str(jpg),
            }
            js.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.root.after(0, self.done, record, full, image_only, jpg, js)
        except Exception:
            self.root.after(0, self.failed, traceback.format_exc())

    def done(self, r, full, image_only, jpg, js):
        self.result = r
        self.full_render = full
        self.image_only = image_only
        self.display_image = full if self.show_full_render.get() else image_only
        self.busy = False
        ctx = r["context"]
        fm = r["foreign_material"]
        fo = (
            "FOREIGN MATERIAL DETECTED"
            if fm["foreign_material_present"]
            else "NO FOREIGN MATERIAL DETECTED"
        )
        cat = f" | {r['diagnostic_category']}" if r["diagnostic_category"] else ""
        self.headline.set(
            f"Context: {ctx['class_name'].upper()} ({ctx['confidence']:.1%}) | "
            f"{fo} | verified={len(fm['detections'])}{cat}"
        )
        lines = [
            fo,
            f"Context: {ctx['class_name']} | confidence={ctx['confidence']:.4f}",
            f"Context GT: {r['context_ground_truth'] or 'not available'}",
            "",
            f"Mode: {'GT comparison' if r['foreign_gt_label'] else 'unlabeled inference'}",
            f"Category: {r['diagnostic_category'] or 'n/a'}",
            f"TP / FP / FN: {r['tp']} / {r['fp']} / {r['fn']}",
            f"Verified FO objects: {len(fm['detections'])}",
            f"Class-uncertain: {r['class_uncertain_verified_objects']}",
            "",
        ]
        for i, d in enumerate(fm["detections"], 1):
            lines += [
                f"Object {i}: {d['class_name'] or 'ForeignMaterial (class uncertain)'}",
                f"  Det={d['detector_confidence']:.3f}  "
                f"Ver={d['verifier_fo_probability']:.3f}  "
                f"Cls={d['exact_confidence']:.3f}",
            ]
        lines += [
            "",
            "Models:",
            f"  Stage 0: {r['models_used']['context']}",
            f"  Stage A: {r['models_used']['stage_a']}",
            f"  Stage B: {r['models_used']['stage_b']}",
            f"  Stage C: {r['models_used']['stage_c']}",
            f"  OP: {r['models_used']['operating_point']}",
            "",
            "Saved:",
            str(jpg),
            str(js),
        ]
        self.set_text("\n".join(lines))
        self.status.set("Analysis complete.")
        self.refresh_image()

    def failed(self, e):
        self.busy = False
        self.status.set("Analysis failed.")
        self.set_text(e)
        messagebox.showerror("Analysis failed", e)

    def refresh_image(self):
        if self.result is not None and self.full_render is not None:
            self.display_image = (
                self.full_render if self.show_full_render.get() else self.image_only
            )
        if self.display_image is None:
            return
        w = max(100, self.canvas.winfo_width() - 12)
        h = max(100, self.canvas.winfo_height() - 12)
        im = self.display_image.copy()
        im.thumbnail((w, h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.photo,
            anchor="center",
        )

    def set_text(self, s):
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", s)
        self.text.config(state="disabled")

    def save_as(self):
        if self.full_render is None:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if p:
            self.full_render.save(p, quality=95)
            self.status.set(f"Saved: {p}")

    def open_output(self):
        p = fmt(self.c["app"]["output_root"], self.T)
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(p)
        except Exception as exc:
            messagebox.showerror("Open output folder", str(exc))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=default_config())
    a = p.parse_args()
    root = tk.Tk()
    ico = bundled(Path("assets") / "CCBRE_AI_LAB_Logo.ico")
    if ico.is_file():
        try:
            root.iconbitmap(default=str(ico))
        except Exception:
            pass
    App(root, a.config)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
