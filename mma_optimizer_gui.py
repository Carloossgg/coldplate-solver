"""
mma_optimizer_gui.py — Desktop GUI for MMA Topology Optimization

Features:
    - New Run : start optimization from current ExportFiles geometry
    - Resume  : load a .npz checkpoint and continue where you left off
    - Stop & Save : cleanly stop and save a checkpoint immediately
    - Save Now : save a checkpoint at any time (even mid-run)

    - Live geometry preview (updates each iteration)
    - Convergence plots (f0 and vol_frac vs iteration)
    - Scrollable console log (redirects stdout/stderr)
    - Editable config parameters

Usage:
    python mma_optimizer_gui.py
"""

from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np

# ---------------------------------------------------------------------------
# We need to be in the project root so that mma_optimizer can import
# run_solvers etc.
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    _THIS_DIR = Path(sys.executable).parent.resolve()
else:
    _THIS_DIR = Path(__file__).parent.resolve()
os.chdir(_THIS_DIR)
sys.path.insert(0, str(_THIS_DIR))

import mma_optimizer as _mma  # noqa: E402  (after sys.path adjustment)
from mma_optimizer import (  # noqa: E402
    OptConfig,
    load_checkpoint,
    save_checkpoint,
    run_optimization,
)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =========================================================================
# STDOUT REDIRECTOR  (feeds optimizer print() into the GUI log)
# =========================================================================
class _StdoutRedirector(io.TextIOBase):
    """Captures writes and posts them to a queue for the GUI to display."""

    def __init__(self, q: queue.Queue):
        self._q = q
        self._original = sys.stdout

    def write(self, text: str) -> int:
        if text:
            self._q.put(("log", text))
            # Also forward to real stdout so VS Code terminal still sees it
            try:
                self._original.write(text)
            except Exception:
                pass
        return len(text)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass


# =========================================================================
# PARAMETER DEFINITIONS
# =========================================================================
_PARAMS = [
    ("max_iter",         "Max Iterations",       int,   100,  {"min": 1,    "max": 1000}),
    ("vol_frac_target",  "Volume Fraction",       float, 0.40, {"min": 0.05, "max": 0.95}),
    ("w_peak",           "Weight: Peak Temp",     float, 1.0,  {"min": 0.0,  "max": 10.0}),
    ("w_uniformity",     "Weight: Uniformity",    float, 0.0,  {"min": 0.0,  "max": 10.0}),
    ("move_limit",       "MMA Move Limit",        float, 0.2,  {"min": 0.01, "max": 1.0}),
    ("filter_radius",    "Filter Radius (cells)", float, 3.0,  {"min": 0.0,  "max": 20.0}),
    ("change_tol",       "Convergence Tol.",      float, 0.01, {"min": 1e-6, "max": 0.5}),
]


# =========================================================================
# MAIN GUI APPLICATION
# =========================================================================
class MMAOptimizerGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MMA Cold-Plate Optimizer")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)

        # ---- configure dark-ish theme ----
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._setup_styles()

        # ---- state ----
        self._q: queue.Queue = queue.Queue()
        self._stop_event: threading.Event | None = None
        self._opt_thread: threading.Thread | None = None
        self._running = False

        # For resume
        self._resume_checkpoint: Path | None = None
        self._resume_data: tuple | None = None  # (gamma, history, mma_state, config, start_iter)

        # ---- build UI ----
        self._build_menu()
        self._build_layout()

        # ---- poll queue every 100 ms ----
        self.root.after(100, self._poll_queue)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # STYLES
    # ------------------------------------------------------------------
    def _setup_styles(self):
        BG = "#1e1e2e"
        PANEL = "#2a2a3e"
        ACCENT = "#7c3aed"
        FG = "#e0e0f0"
        ENTRY_BG = "#12121c"

        self.root.configure(bg=BG)

        self.style.configure(".", background=BG, foreground=FG, font=("Segoe UI", 9))
        self.style.configure("TFrame", background=BG)
        self.style.configure("Panel.TFrame", background=PANEL)
        self.style.configure("TLabel", background=BG, foreground=FG)
        self.style.configure("Panel.TLabel", background=PANEL, foreground=FG)
        self.style.configure("Header.TLabel", background=BG, foreground=ACCENT,
                              font=("Segoe UI", 10, "bold"))
        self.style.configure("PanelHeader.TLabel", background=PANEL,
                              foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        self.style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=FG,
                              insertcolor=FG)
        self.style.configure("TButton", background=PANEL, foreground=FG,
                              font=("Segoe UI", 9, "bold"), padding=5)
        self.style.map("TButton",
                       background=[("active", ACCENT)],
                       foreground=[("active", "#ffffff")])
        self.style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff")
        self.style.map("Accent.TButton",
                       background=[("active", "#6d28d9")])
        self.style.configure("Danger.TButton", background="#7f1d1d", foreground="#fecaca")
        self.style.map("Danger.TButton",
                       background=[("active", "#991b1b")])
        self.style.configure("TLabelframe", background=PANEL, foreground=ACCENT)
        self.style.configure("TLabelframe.Label", background=PANEL, foreground=ACCENT,
                              font=("Segoe UI", 9, "bold"))
        self.style.configure("Horizontal.TProgressbar",
                              troughcolor=ENTRY_BG, background=ACCENT)
        self.style.configure("TSeparator", background=ACCENT)

        self._colors = {
            "BG": BG, "PANEL": PANEL, "ACCENT": ACCENT,
            "FG": FG, "ENTRY_BG": ENTRY_BG,
        }

    # ------------------------------------------------------------------
    # MENU
    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root, tearoff=0,
                          bg=self._colors["PANEL"], fg=self._colors["FG"],
                          activebackground=self._colors["ACCENT"])
        file_menu = tk.Menu(menubar, tearoff=0,
                             bg=self._colors["PANEL"], fg=self._colors["FG"],
                             activebackground=self._colors["ACCENT"])
        file_menu.add_command(label="Open Output Folder",
                              command=self._open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0,
                            bg=self._colors["PANEL"], fg=self._colors["FG"],
                            activebackground=self._colors["ACCENT"])
        view_menu.add_command(label="Geometry Animation…",
                              command=self._cmd_show_animation)
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

    def _cmd_show_animation(self):
        """Open the geometry animation viewer window."""
        iters_dir = _THIS_DIR / "ExportFiles" / "iterations"
        if not iters_dir.exists():
            messagebox.showinfo(
                "No Data",
                "No iterations folder found.\n"
                "Run at least one optimization iteration first.",
            )
            return
        _GeometryAnimationWindow(self.root, iters_dir, self._colors)


    # ------------------------------------------------------------------
    # LAYOUT
    # ------------------------------------------------------------------
    def _build_layout(self):
        C = self._colors

        # ---- Top toolbar ----
        toolbar = ttk.Frame(self.root, style="Panel.TFrame")
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)
        self._build_toolbar(toolbar)

        # ---- Status bar ----
        status_bar = ttk.Frame(self.root, style="Panel.TFrame", height=28)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(status_bar, textvariable=self._status_var,
                  style="Panel.TLabel").pack(side=tk.LEFT, padx=8, pady=4)

        # ---- Main area (left + right panes) ----
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left pane: config + progress
        left_frame = ttk.Frame(main, style="TFrame", width=280)
        left_frame.pack_propagate(False)
        main.add(left_frame, weight=0)

        self._build_config_panel(left_frame)
        self._build_progress_panel(left_frame)

        # Right pane: plots + log
        right_frame = ttk.Frame(main, style="TFrame")
        main.add(right_frame, weight=1)

        self._build_right_panel(right_frame)

    def _build_toolbar(self, parent):
        C = self._colors

        # Title
        ttk.Label(parent, text="⚙  MMA Optimizer",
                  style="PanelHeader.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=12, pady=6)

        # Separator
        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                        padx=6, pady=4)

        # Buttons
        self._btn_new = ttk.Button(parent, text="▶  New Run",
                                   style="Accent.TButton",
                                   command=self._cmd_new_run)
        self._btn_new.pack(side=tk.LEFT, padx=4, pady=6)

        self._btn_resume = ttk.Button(parent, text="⟳  Resume",
                                      command=self._cmd_resume)
        self._btn_resume.pack(side=tk.LEFT, padx=4, pady=6)

        self._btn_stop = ttk.Button(parent, text="■  Stop & Save",
                                    style="Danger.TButton",
                                    command=self._cmd_stop_save,
                                    state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=4, pady=6)

        self._btn_savenow = ttk.Button(parent, text="💾  Save Now",
                                       command=self._cmd_save_now,
                                       state=tk.DISABLED)
        self._btn_savenow.pack(side=tk.LEFT, padx=4, pady=6)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                        padx=6, pady=4)

        self._btn_clear_log = ttk.Button(parent, text="Clear Log",
                                         command=self._clear_log)
        self._btn_clear_log.pack(side=tk.LEFT, padx=4, pady=6)

    def _build_config_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="Configuration", padding=8)
        lf.pack(fill=tk.X, padx=6, pady=6)

        self._param_vars: dict[str, tk.Variable] = {}

        for attr, label, typ, default, _ in _PARAMS:
            row = ttk.Frame(lf, style="TFrame")
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=22, anchor="w").pack(side=tk.LEFT)
            var = tk.IntVar(value=default) if typ is int else tk.DoubleVar(value=default)
            entry = ttk.Entry(row, textvariable=var, width=10)
            entry.pack(side=tk.LEFT, padx=4)
            self._param_vars[attr] = var

        # Resume file indicator
        self._resume_label_var = tk.StringVar(value="No checkpoint selected")
        ttk.Label(lf, textvariable=self._resume_label_var,
                  foreground="#a78bfa", wraplength=240,
                  font=("Segoe UI", 8)).pack(fill=tk.X, pady=(6, 0))

    def _build_progress_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="Progress", padding=8)
        lf.pack(fill=tk.X, padx=6, pady=2)

        # Iteration label
        self._iter_var = tk.StringVar(value="Iteration: —")
        ttk.Label(lf, textvariable=self._iter_var).pack(anchor="w")

        # Objective label
        self._f0_var = tk.StringVar(value="Objective (f0): —")
        ttk.Label(lf, textvariable=self._f0_var).pack(anchor="w")

        # Vol frac label
        self._vf_var = tk.StringVar(value="Vol. Fraction: —")
        ttk.Label(lf, textvariable=self._vf_var).pack(anchor="w")

        # Max change label
        self._chg_var = tk.StringVar(value="Max Change: —")
        ttk.Label(lf, textvariable=self._chg_var).pack(anchor="w")

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            lf, variable=self._progress_var, maximum=100,
            style="Horizontal.TProgressbar",
        )
        self._progress_bar.pack(fill=tk.X, pady=(6, 0))

    def _build_right_panel(self, parent):
        C = self._colors

        # Vertical PanedWindow (top=plots, bottom=log)
        vpane = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        vpane.pack(fill=tk.BOTH, expand=True)

        # ---- Top: two plot frames side by side ----
        plot_frame = ttk.Frame(vpane, style="TFrame", height=300)
        vpane.add(plot_frame, weight=1)

        # Geometry preview (left)
        geom_lf = ttk.LabelFrame(plot_frame, text="Geometry Preview", padding=4)
        geom_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        # Convergence plot (right)
        conv_lf = ttk.LabelFrame(plot_frame, text="Convergence", padding=4)
        conv_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))

        if HAS_MATPLOTLIB:
            self._build_geometry_plot(geom_lf)
            self._build_convergence_plot(conv_lf)
        else:
            ttk.Label(geom_lf, text="matplotlib not available").pack(expand=True)
            ttk.Label(conv_lf, text="matplotlib not available").pack(expand=True)

        # ---- Bottom: log console ----
        log_lf = ttk.LabelFrame(vpane, text="Console Output", padding=4, height=250)
        vpane.add(log_lf, weight=0)

        self._log = scrolledtext.ScrolledText(
            log_lf,
            bg=self._colors["ENTRY_BG"],
            fg=self._colors["FG"],
            font=("Cascadia Code", 8) if _font_exists("Cascadia Code") else ("Courier New", 8),
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            insertbackground=self._colors["FG"],
        )
        self._log.pack(fill=tk.BOTH, expand=True)

    def _build_geometry_plot(self, parent):
        fig_g, ax_g = plt.subplots(figsize=(5, 2.5),
                                   facecolor=self._colors["PANEL"])
        ax_g.set_facecolor(self._colors["ENTRY_BG"])
        ax_g.set_title("Waiting for data…", color=self._colors["FG"], fontsize=8)
        ax_g.tick_params(colors=self._colors["FG"])
        for spine in ax_g.spines.values():
            spine.set_edgecolor(self._colors["ACCENT"])

        canvas_g = FigureCanvasTkAgg(fig_g, master=parent)
        canvas_g.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas_g.draw()

        self._fig_geom = fig_g
        self._ax_geom = ax_g
        self._canvas_geom = canvas_g
        self._im_geom = None   # imshow handle
        self._cbar_geom = None # colorbar handle

    def _build_convergence_plot(self, parent):
        fig_c, axes = plt.subplots(3, 1, figsize=(5, 3.5), sharex=True,
                                   facecolor=self._colors["PANEL"])
        for ax in axes:
            ax.set_facecolor(self._colors["ENTRY_BG"])
            ax.tick_params(colors=self._colors["FG"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(self._colors["ACCENT"])

        axes[0].set_ylabel("f0", color=self._colors["FG"], fontsize=8)
        axes[1].set_ylabel("vol frac", color=self._colors["FG"], fontsize=8)
        axes[2].set_ylabel("T_peak (°C)", color=self._colors["FG"], fontsize=8)
        axes[2].set_xlabel("Iteration", color=self._colors["FG"], fontsize=8)
        fig_c.tight_layout(pad=0.5)

        canvas_c = FigureCanvasTkAgg(fig_c, master=parent)
        canvas_c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas_c.draw()

        self._fig_conv = fig_c
        self._axes_conv = axes
        self._canvas_conv = canvas_c

    # ------------------------------------------------------------------
    # TOOLBAR COMMANDS
    # ------------------------------------------------------------------
    def _cmd_new_run(self):
        if self._running:
            messagebox.showwarning("Running", "Stop the current run first.")
            return
        config = self._build_config()
        self._start_run(config, start_iter=0)

    def _cmd_resume(self):
        if self._running:
            messagebox.showwarning("Running", "Stop the current run first.")
            return

        export_dir = _THIS_DIR / "ExportFiles"
        init_dir = str(export_dir) if export_dir.exists() else str(_THIS_DIR)

        path = filedialog.askopenfilename(
            title="Select checkpoint (.npz)",
            initialdir=init_dir,
            filetypes=[("Checkpoint files", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            gamma, history, mma_state, config, start_iter = load_checkpoint(path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load checkpoint:\n{e}")
            return

        # Update GUI config fields to match checkpoint
        for attr, _, typ, default, _ in _PARAMS:
            val = getattr(config, attr, default)
            if attr in self._param_vars:
                try:
                    self._param_vars[attr].set(val)
                except Exception:
                    pass

        self._resume_label_var.set(f"Checkpoint: {Path(path).name}  (iter {start_iter - 1})")
        self._log_write(f"\n[Resume] Loaded checkpoint: {path}\n")
        self._log_write(f"[Resume] Will start from iteration {start_iter}\n\n")

        # ---- Immediately preview geometry and history plots ----
        if HAS_MATPLOTLIB and gamma is not None:
            last_f0 = history["f0"][-1] if history.get("f0") else float("nan")
            last_vf = history["vol_frac"][-1] if history.get("vol_frac") else float("nan")
            self._update_geometry_plot(gamma, start_iter - 1, last_f0, last_vf)
        if HAS_MATPLOTLIB and history:
            self._update_convergence_plot(history)

        # Update progress labels from history
        if history.get("iterations"):
            it_prev = history["iterations"][-1]
            self._iter_var.set(f"Iteration: {it_prev}  (checkpoint)")
        if history.get("f0"):
            self._f0_var.set(f"Objective (f0): {history['f0'][-1]:.6f}")
        if history.get("vol_frac"):
            self._vf_var.set(f"Vol. Fraction: {history['vol_frac'][-1]:.4f} (target: {config.vol_frac_target:.2f})")
        pct = 100.0 * start_iter / max(config.max_iter, 1)
        self._progress_var.set(min(pct, 100.0))

        # Update max_iter if checkpoint had fewer remaining iters
        if start_iter >= config.max_iter:
            new_max = start_iter + int(self._param_vars["max_iter"].get())
            config.max_iter = new_max
            self._param_vars["max_iter"].set(new_max)

        self._start_run(
            config,
            start_iter=start_iter,
            gamma_init=gamma,
            history_init=history,
            mma_state_init=mma_state,
        )

    def _cmd_stop_save(self):
        if not self._running or self._stop_event is None:
            return
        self._log_write("\n[GUI] Stop & Save requested — will stop after current iteration completes.\n")
        self._stop_event.set()
        self._set_running(False)

    def _cmd_save_now(self):
        """Save a checkpoint immediately by pickling the last-known state."""
        if not hasattr(self, "_last_snap"):
            messagebox.showinfo("No data", "No iteration data available yet.")
            return

        snap = self._last_snap
        export_dir = _THIS_DIR / "ExportFiles"
        export_dir.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save checkpoint as…",
            initialdir=str(export_dir),
            initialfile=f"checkpoint_iter_{snap['iteration']:04d}.npz",
            defaultextension=".npz",
            filetypes=[("Checkpoint files", "*.npz")],
        )
        if not path:
            return
        try:
            save_checkpoint(
                path,
                snap["gamma"],
                snap["history"],
                snap["mma_state"],
                snap["config"],
                snap["iteration"],
            )
            self._log_write(f"[GUI] Checkpoint saved to {path}\n")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save:\n{e}")

    def _clear_log(self):
        self._log.configure(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.configure(state=tk.DISABLED)

    def _open_output_folder(self):
        folder = _THIS_DIR / "ExportFiles"
        folder.mkdir(exist_ok=True)
        import subprocess
        subprocess.Popen(["explorer", str(folder)])

    # ------------------------------------------------------------------
    # RUN MANAGEMENT
    # ------------------------------------------------------------------
    def _build_config(self) -> OptConfig:
        config = OptConfig()
        for attr, _, typ, default, _ in _PARAMS:
            try:
                val = typ(self._param_vars[attr].get())
            except Exception:
                val = default
            setattr(config, attr, val)
        return config

    def _start_run(
        self,
        config: OptConfig,
        start_iter: int = 0,
        gamma_init=None,
        history_init=None,
        mma_state_init=None,
    ):
        self._stop_event = threading.Event()
        self._last_config = config  # keep reference for save-now

        # Clear any previous last-snap that belonged to a different run
        if start_iter == 0:
            self._last_snap = None  # type: ignore[assignment]

        export_dir = _THIS_DIR / "ExportFiles"
        export_dir.mkdir(exist_ok=True)

        # Redirect stdout → GUI log
        self._stdout_redirector = _StdoutRedirector(self._q)
        sys.stdout = self._stdout_redirector

        def _thread_fn():
            try:
                run_optimization(
                    config=config,
                    start_iter=start_iter,
                    gamma_init=gamma_init,
                    history_init=history_init,
                    mma_state_init=mma_state_init,
                    stop_event=self._stop_event,
                    on_progress=self._on_progress,
                    checkpoint_dir=export_dir,
                )
            except Exception as exc:
                import traceback
                self._q.put(("log", f"\n[ERROR] {exc}\n{traceback.format_exc()}\n"))
            finally:
                sys.stdout = sys.__stdout__
                self._q.put(("done", None))

        self._opt_thread = threading.Thread(target=_thread_fn, daemon=True)
        self._opt_thread.start()
        self._set_running(True)

    def _on_progress(self, iteration, f0, vol_frac, max_change, gamma, history):
        """Called from optimizer thread — post to queue for GUI update."""
        # We need the mma_state to save checkpoints from GUI.
        # We capture it by looking it up inside the running optimizer
        # via a closure trick: the optimizer calls this after updating mma_state,
        # and we store all we need to reconstruct a checkpoint.
        self._q.put(("progress", {
            "iteration": iteration,
            "f0": f0,
            "vol_frac": vol_frac,
            "max_change": max_change,
            "gamma": gamma,
            "history": history,
            "config": self._last_config,
        }))

    def _set_running(self, running: bool):
        self._running = running
        state_run = tk.DISABLED if running else tk.NORMAL
        state_stop = tk.NORMAL if running else tk.DISABLED
        self._btn_new.configure(state=state_run)
        self._btn_resume.configure(state=state_run)
        self._btn_stop.configure(state=state_stop)
        self._btn_savenow.configure(state=state_stop)
        status = "⚙ Optimization running…" if running else "Ready"
        self._status_var.set(status)

    # ------------------------------------------------------------------
    # QUEUE POLLING  (called from main thread every 100 ms)
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self._q.get_nowait()
                if msg_type == "log":
                    self._log_write(payload)
                elif msg_type == "progress":
                    self._handle_progress(payload)
                elif msg_type == "done":
                    self._handle_done()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _log_write(self, text: str):
        self._log.configure(state=tk.NORMAL)
        self._log.insert(tk.END, text)
        self._log.see(tk.END)
        self._log.configure(state=tk.DISABLED)

    def _handle_progress(self, data: dict):
        it = data["iteration"]
        f0 = data["f0"]
        vf = data["vol_frac"]
        chg = data["max_change"]
        gamma = data["gamma"]
        history = data["history"]
        config = data["config"]

        # Store latest snapshot for "Save Now"
        self._last_snap = {
            "iteration": it,
            "f0": f0,
            "gamma": gamma,
            "history": history,
            "config": config,
            # mma_state not easily available here; save_now will use partial info
            "mma_state": _DummyMMAState(it),
        }

        # Update progress labels
        self._iter_var.set(f"Iteration: {it}")
        self._f0_var.set(f"Objective (f0): {f0:.6f}")
        self._vf_var.set(f"Vol. Fraction: {vf:.4f} (target: {config.vol_frac_target:.2f})")
        self._chg_var.set(f"Max Change: {chg:.6f}")

        # Progress bar (based on max_iter)
        pct = 100.0 * (it + 1) / max(config.max_iter, 1)
        self._progress_var.set(min(pct, 100.0))

        # Update plots
        if HAS_MATPLOTLIB:
            self._update_geometry_plot(gamma, it, f0, vf)
            self._update_convergence_plot(history)

    def _handle_done(self):
        sys.stdout = sys.__stdout__
        self._set_running(False)
        self._log_write("\n[GUI] Optimization finished.\n")
        self._status_var.set("Finished")

    # ------------------------------------------------------------------
    # PLOT UPDATES
    # ------------------------------------------------------------------
    def _update_geometry_plot(self, gamma: np.ndarray, it: int, f0: float, vf: float):
        ax = self._ax_geom
        fig = self._fig_geom
        ax.clear()
        # Remove old colorbar axes so layout stays stable
        if self._cbar_geom is not None:
            try:
                self._cbar_geom.remove()
            except Exception:
                pass
            self._cbar_geom = None
        ax.set_facecolor(self._colors["ENTRY_BG"])
        im = ax.imshow(gamma, cmap="gray", vmin=0, vmax=1,
                       aspect="auto", origin="upper")
        ax.set_title(
            f"Iter {it}  |  f0={f0:.4f}  |  vol={vf:.3f}",
            color=self._colors["FG"], fontsize=8
        )
        ax.tick_params(colors=self._colors["FG"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(self._colors["ACCENT"])
        # ---- Colorbar legend (0 = Solid, 1 = Liquid) ----
        cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                            fraction=0.046, pad=0.04)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["0  Solid", "1  Liquid"])
        cbar.ax.yaxis.set_tick_params(color=self._colors["FG"], labelsize=7)
        cbar.outline.set_edgecolor(self._colors["ACCENT"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=self._colors["FG"])
        self._cbar_geom = cbar
        fig.tight_layout(pad=0.3)
        self._canvas_geom.draw_idle()

    def _update_convergence_plot(self, history: dict):
        iters = history.get("iterations", [])
        f0s = history.get("f0", [])
        vfs = history.get("vol_frac", [])
        t_peaks = history.get("T_peak", [])

        if not iters:
            return

        ax0, ax1, ax2 = self._axes_conv
        FG = self._colors["FG"]
        ACCENT = self._colors["ACCENT"]

        ax0.clear()
        ax0.set_facecolor(self._colors["ENTRY_BG"])
        ax0.plot(iters, f0s, color=ACCENT, lw=1.5)
        ax0.set_ylabel("f0", color=FG, fontsize=8)
        ax0.tick_params(colors=FG, labelsize=7)
        for spine in ax0.spines.values():
            spine.set_edgecolor(ACCENT)

        ax1.clear()
        ax1.set_facecolor(self._colors["ENTRY_BG"])
        ax1.plot(iters, vfs, color="#34d399", lw=1.5)
        ax1.set_ylabel("vol frac", color=FG, fontsize=8)
        ax1.tick_params(colors=FG, labelsize=7)
        for spine in ax1.spines.values():
            spine.set_edgecolor(ACCENT)

        ax2.clear()
        ax2.set_facecolor(self._colors["ENTRY_BG"])
        t_peaks_clipped = [max(20.0, min(200.0, v)) for v in t_peaks]
        ax2.plot(iters, t_peaks_clipped, color="#f87171", lw=1.5)
        ax2.set_ylim(20.0, 200.0)
        ax2.set_ylabel("T_peak (°C)", color=FG, fontsize=8)
        ax2.set_xlabel("Iteration", color=FG, fontsize=8)
        ax2.tick_params(colors=FG, labelsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor(ACCENT)

        self._fig_conv.tight_layout(pad=0.5)
        self._canvas_conv.draw_idle()

    # ------------------------------------------------------------------
    # WINDOW CLOSE
    # ------------------------------------------------------------------
    def _on_close(self):
        if self._running:
            if not messagebox.askyesno(
                "Quit",
                "Optimization is running. Stop it and quit? "
                "(The current iteration state will NOT be auto-saved.)",
            ):
                return
            if self._stop_event:
                self._stop_event.set()
            # Hard-kill any solver subprocess that is currently running
            # (simple.exe / thermal_solver.exe won't stop on their own)
            try:
                import run_solvers as _rs
                _rs.kill_active_process()
            except Exception:
                pass
        sys.stdout = sys.__stdout__
        self.root.destroy()
        # Hard-exit the Python process so no background threads or
        # lingering state can keep the process alive after the GUI closes.
        import os as _os
        _os._exit(0)



# =========================================================================
# GEOMETRY ANIMATION WINDOW
# =========================================================================
class _GeometryAnimationWindow:
    """
    Toplevel window that plays an animation of geometry frames
    loaded from ExportFiles/iterations/iter*/gamma_iter_*.txt.
    """

    def __init__(self, parent: tk.Tk, iters_dir: Path, colors: dict):
        self._colors = colors
        self._frames: list[tuple[int, np.ndarray]] = []  # (iter_number, gamma)
        self._frame_idx = 0
        self._playing = False
        self._after_id = None

        # --- Scan and load frames ---
        self._load_frames(iters_dir)
        if not self._frames:
            messagebox.showinfo(
                "No Frames",
                "No gamma_iter_*.txt files found in the iterations folder.\n"
                "Run some optimization iterations first.",
                parent=parent,
            )
            return

        # --- Build window ---
        self._win = tk.Toplevel(parent)
        self._win.title("Geometry Animation")
        self._win.geometry("900x540")
        self._win.configure(bg=colors["BG"])
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._show_frame(0)

    # ------------------------------------------------------------------
    def _load_frames(self, iters_dir: Path):
        """Scan iter* subfolders for gamma_iter_*.txt files."""
        import re
        entries: list[tuple[int, Path]] = []
        for iter_folder in sorted(iters_dir.iterdir()):
            if not iter_folder.is_dir():
                continue
            for fpath in iter_folder.glob("gamma_iter_*.txt"):
                m = re.search(r"gamma_iter_(\d+)", fpath.name)
                if m:
                    entries.append((int(m.group(1)), fpath))
        # Sort by iteration number
        entries.sort(key=lambda x: x[0])
        for it_num, fpath in entries:
            try:
                gamma = np.loadtxt(str(fpath))
                self._frames.append((it_num, gamma))
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _build_ui(self):
        C = self._colors
        n = len(self._frames)

        # ---- matplotlib canvas ----
        if not HAS_MATPLOTLIB:
            tk.Label(self._win, text="matplotlib not available",
                     bg=C["BG"], fg=C["FG"]).pack(expand=True)
            return

        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=C["PANEL"])
        ax.set_facecolor(C["ENTRY_BG"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["ACCENT"])

        canvas = FigureCanvasTkAgg(fig, master=self._win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 2))

        self._fig = fig
        self._ax = ax
        self._canvas = canvas
        self._im = None

        # ---- Controls ----
        ctrl = tk.Frame(self._win, bg=C["PANEL"], pady=6)
        ctrl.pack(fill=tk.X, padx=6, pady=(0, 6))

        # Play / Pause
        self._play_btn = tk.Button(
            ctrl, text="▶  Play",
            bg=C["ACCENT"], fg="#ffffff", font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT, padx=10,
            command=self._toggle_play,
        )
        self._play_btn.pack(side=tk.LEFT, padx=(8, 4))

        # Frame label
        self._frame_lbl = tk.Label(
            ctrl, text=f"Frame 1 / {n}  (iter 0)",
            bg=C["PANEL"], fg=C["FG"], font=("Segoe UI", 9),
        )
        self._frame_lbl.pack(side=tk.LEFT, padx=8)

        # Speed
        tk.Label(ctrl, text="Speed:", bg=C["PANEL"], fg=C["FG"],
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 2))
        self._speed_var = tk.IntVar(value=200)   # ms per frame
        speed_scale = tk.Scale(
            ctrl, from_=50, to=1000, resolution=50,
            orient=tk.HORIZONTAL, variable=self._speed_var,
            label="", showvalue=True,
            bg=C["PANEL"], fg=C["FG"], troughcolor=C["ENTRY_BG"],
            highlightthickness=0, length=140,
        )
        speed_scale.pack(side=tk.LEFT)
        tk.Label(ctrl, text="ms/frame", bg=C["PANEL"], fg=C["FG"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 12))

        # Frame slider
        slider_frame = tk.Frame(self._win, bg=C["BG"])
        slider_frame.pack(fill=tk.X, padx=6, pady=(0, 4))

        self._slider_var = tk.IntVar(value=0)
        self._slider = tk.Scale(
            slider_frame, from_=0, to=n - 1,
            orient=tk.HORIZONTAL, variable=self._slider_var,
            label="", showvalue=False,
            bg=C["BG"], fg=C["FG"], troughcolor=C["ENTRY_BG"],
            highlightthickness=0,
            command=self._on_slider,
        )
        self._slider.pack(fill=tk.X, padx=4)

    # ------------------------------------------------------------------
    def _show_frame(self, idx: int):
        idx = max(0, min(idx, len(self._frames) - 1))
        self._frame_idx = idx
        it_num, gamma = self._frames[idx]

        if self._im is None:
            self._im = self._ax.imshow(
                gamma, cmap="gray", vmin=0, vmax=1,
                aspect="auto", origin="upper",
            )
        else:
            self._im.set_data(gamma)

        self._ax.set_title(
            f"Iteration {it_num}  |  frame {idx + 1} / {len(self._frames)}",
            color=self._colors["FG"], fontsize=10,
        )
        self._canvas.draw_idle()

        # Update controls
        self._slider_var.set(idx)
        self._frame_lbl.config(
            text=f"Frame {idx + 1} / {len(self._frames)}  (iter {it_num})"
        )

    # ------------------------------------------------------------------
    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._play_btn.config(text="▶  Play")
            if self._after_id:
                self._win.after_cancel(self._after_id)
                self._after_id = None
        else:
            self._playing = True
            self._play_btn.config(text="⏸  Pause")
            self._schedule_next()

    def _schedule_next(self):
        if not self._playing:
            return
        delay = int(self._speed_var.get())
        self._after_id = self._win.after(delay, self._advance)

    def _advance(self):
        if not self._playing:
            return
        next_idx = self._frame_idx + 1
        if next_idx >= len(self._frames):
            next_idx = 0   # loop
        self._show_frame(next_idx)
        self._schedule_next()

    def _on_slider(self, val):
        idx = int(float(val))
        if idx != self._frame_idx:
            if self._playing:
                self._toggle_play()  # pause when user drags
            self._show_frame(idx)

    def _on_close(self):
        if self._after_id:
            self._win.after_cancel(self._after_id)
        self._playing = False
        plt.close(self._fig)
        self._win.destroy()


# =========================================================================
# HELPERS
# =========================================================================
class _DummyMMAState:
    """Minimal MMAState stub for save_checkpoint when full state is unavailable."""
    def __init__(self, iter_count: int):
        self.iter = iter_count
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None


def _font_exists(name: str) -> bool:
    try:
        import tkinter.font as tkfont
        return name in tkfont.families()
    except Exception:
        return False


# =========================================================================
# ENTRY POINT
# =========================================================================
def main():
    root = tk.Tk()
    app = MMAOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
