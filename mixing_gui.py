"""
Interactive front end for the mixing solvers.

Run with

    python mixing_gui.py

Pick a mode, set the grid and the initial conditions, press RUN.  The tabs on
the right let you scrub through the recorded snapshots; every run also writes
its figures to the output folder, so the pictures in the paper can be produced
either from here or from the command line in mixing.py.

All of the numerics live in mixing.py -- this file is only the viewer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")  # must happen before the canvas import below

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import mixing as mx

PLAY_DELAY_MS = 180  # frame delay when playing a run back

ARROW_OPEN = "\u25bc"
ARROW_CLOSED = "\u25b6"
WHEEL_EVENTS = ("<MouseWheel>", "<Button-4>", "<Button-5>")


def open_folder(path):
    """Show a directory in the platform's file browser."""
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def status_line(data: dict, snapshot: dict, index: int, count: int) -> str:
    """One-line summary of where we are in the run and how healthy it looks."""
    parts = [
        f"snapshot {index + 1}/{count}",
        f"step {snapshot['step']}",
        f"t = {snapshot['time']:.4f}",
        f"E = {snapshot['energy']:.6e}",
        f"max|div v| = {snapshot['divergence_spectral_linf']:.2e}",
        f"max|rho - rho(0)| = {snapshot['rho_change_linf']:.3e}",
    ]
    return "     ".join(parts)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


class SimpleTab:
    """A tab holding one static matplotlib figure."""

    def __init__(self, parent, title, fig):
        self.frame = ttk.Frame(parent)
        parent.add(self.frame, text=title)

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

    def destroy(self):
        self.toolbar.destroy()
        self.canvas.get_tk_widget().destroy()
        self.frame.destroy()


class TextTab:
    """A tab holding fixed-width text, used for the diagnostics report."""

    def __init__(self, parent, title, text):
        self.frame = ttk.Frame(parent)
        parent.add(self.frame, text=title)

        box = tk.Text(self.frame, wrap="none", font=("Courier New", 11))
        box.insert("1.0", text)
        box.configure(state="disabled")
        box.pack(side="left", fill="both", expand=True)

        bar = ttk.Scrollbar(self.frame, orient="vertical", command=box.yview)
        bar.pack(side="right", fill="y")
        box.configure(yscrollcommand=bar.set)

        self.box = box
        self.bar = bar

    def destroy(self):
        self.bar.destroy()
        self.box.destroy()
        self.frame.destroy()


class SnapshotTab:
    """Base class for the tabs that step through the recorded snapshots.

    Everything shared -- the transport controls, the slider, playback, the
    canvas -- lives here.  Subclasses only have to say what to draw
    (`build_panels`) and how to point it at a different snapshot (`refresh`).
    """

    figsize = (13.0, 8.0)

    def __init__(self, parent, title, data, options=None):
        self.frame = ttk.Frame(parent)
        parent.add(self.frame, text=title)

        self.data = data
        self.grid = data["grid"]
        self.snapshots = data["snapshots"]
        self.count = len(self.snapshots)
        self.options = options or mx.quantities_for(data["mode"])

        self.index_var = tk.IntVar(value=0)
        self.playing = False
        self.after_id = None

        controls = ttk.Frame(self.frame, padding=(8, 8, 8, 2))
        controls.pack(fill="x")

        self.status = ttk.Label(controls, text="")
        self.status.pack(anchor="w", pady=(0, 6))

        self.selector_row = ttk.Frame(controls)
        self.selector_row.pack(fill="x", pady=(0, 6))
        self.add_selectors(self.selector_row)

        row = ttk.Frame(controls)
        row.pack(fill="x")

        ttk.Button(row, text="previous", command=self.step_back).pack(side="left", padx=(0, 6))
        self.play_button = ttk.Button(row, text="play", command=self.toggle_play)
        self.play_button.pack(side="left", padx=(0, 6))
        ttk.Button(row, text="next", command=self.step_forward).pack(side="left", padx=(0, 12))

        self.scale = tk.Scale(
            row,
            from_=0,
            to=max(0, self.count - 1),
            orient="horizontal",
            variable=self.index_var,
            showvalue=False,
            resolution=1,
            command=lambda value: self.show(int(round(float(value)))),
        )
        self.scale.pack(side="left", fill="x", expand=True)

        self.fig = Figure(figsize=self.figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

        if self.count == 0:
            self.status.configure(text="no snapshots were recorded")
            self.scale.configure(state="disabled")
        else:
            self.rebuild(0)

    # -- hooks ---------------------------------------------------------------

    def add_selectors(self, parent):
        """Add quantity drop-downs, if this tab has any."""

    def build_panels(self, snapshot):
        raise NotImplementedError

    def refresh(self, snapshot):
        raise NotImplementedError

    # -- shared behaviour ----------------------------------------------------

    def rebuild(self, index=None):
        """Redraw from scratch, e.g. after the user picks another quantity."""
        if self.count == 0:
            return

        index = self.index_var.get() if index is None else index
        index = max(0, min(int(index), self.count - 1))

        self.fig.clear()
        self.build_panels(self.snapshots[index])
        self.show(index)

    def show(self, index):
        if self.count == 0:
            return

        index = max(0, min(int(index), self.count - 1))
        self.index_var.set(index)

        snapshot = self.snapshots[index]
        self.refresh(snapshot)
        self.status.configure(text=status_line(self.data, snapshot, index, self.count))
        self.canvas.draw_idle()

    def step_back(self):
        self.show(self.index_var.get() - 1)

    def step_forward(self):
        self.show(self.index_var.get() + 1)

    def toggle_play(self):
        if self.playing:
            self.stop()
            return

        # Restart from the beginning if we are sitting on the last frame.
        if self.index_var.get() >= self.count - 1:
            self.show(0)

        self.playing = True
        self.play_button.configure(text="pause")
        self.advance()

    def advance(self):
        if not self.playing:
            return

        index = self.index_var.get() + 1

        if index >= self.count:
            self.show(self.count - 1)
            self.stop()
            return

        self.show(index)
        self.after_id = self.frame.after(PLAY_DELAY_MS, self.advance)

    def stop(self):
        self.playing = False
        self.play_button.configure(text="play")

        if self.after_id is not None:
            try:
                self.frame.after_cancel(self.after_id)
            except tk.TclError:
                pass  # the widget is already gone
            self.after_id = None

    def destroy(self):
        self.stop()
        self.toolbar.destroy()
        self.canvas.get_tk_widget().destroy()
        self.frame.destroy()


class OverviewTab(SnapshotTab):
    """Every stage of the velocity pipeline at once."""

    def __init__(self, parent, title, data):
        labels, rows, cols = mx.OVERVIEW_LAYOUT[data["mode"]]
        self.labels = labels
        self.rows = rows
        self.cols = cols
        self.panels = []
        super().__init__(parent, title, data)

    def build_panels(self, snapshot):
        self.panels = []

        for i, label in enumerate(self.labels):
            ax = self.fig.add_subplot(self.rows, self.cols, i + 1)
            self.panels.append(
                mx.FieldPanel(self.fig, ax, self.grid, snapshot, self.options[label])
            )

        self.fig.subplots_adjust(
            left=0.04, right=0.99, bottom=0.06, top=0.92, wspace=0.36, hspace=0.50
        )

    def refresh(self, snapshot):
        for panel in self.panels:
            panel.update(snapshot)


class SingleQuantityTab(SnapshotTab):
    """One quantity at a time, larger, with a 3D view alongside for vectors."""

    figsize = (12.5, 7.4)

    def __init__(self, parent, title, data):
        self.quantity_var = tk.StringVar()
        self.flat_panel = None
        self.surface_panel = None
        super().__init__(parent, title, data)

    def add_selectors(self, parent):
        self.quantity_var.set(next(iter(self.options)))

        ttk.Label(parent, text="quantity").pack(side="left", padx=(0, 6))

        box = ttk.Combobox(
            parent,
            textvariable=self.quantity_var,
            values=list(self.options),
            state="readonly",
            width=34,
        )
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda event: self.rebuild())

    def build_panels(self, snapshot):
        quantity = self.options[self.quantity_var.get()]

        if quantity.kind == "vector":
            ax_flat = self.fig.add_subplot(1, 2, 1)
            ax_surface = self.fig.add_subplot(1, 2, 2, projection="3d")

            self.flat_panel = mx.FieldPanel(
                self.fig, ax_flat, self.grid, snapshot, quantity, colorbar=True
            )
            self.surface_panel = mx.SurfacePanel(
                self.fig, ax_surface, self.grid, snapshot, quantity
            )

            self.fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.88, wspace=0.32)
        else:
            ax = self.fig.add_subplot(1, 1, 1)

            self.flat_panel = mx.FieldPanel(
                self.fig, ax, self.grid, snapshot, quantity, colorbar=True
            )
            self.surface_panel = None

            self.fig.subplots_adjust(left=0.10, right=0.92, bottom=0.10, top=0.88)

    def refresh(self, snapshot):
        self.flat_panel.update(snapshot)

        if self.surface_panel is not None:
            self.surface_panel.update(snapshot)


class CompareTab(SnapshotTab):
    """Two quantities side by side as 3D surfaces."""

    figsize = (12.5, 7.6)

    def __init__(self, parent, title, data):
        self.left_var = tk.StringVar()
        self.right_var = tk.StringVar()
        self.left_panel = None
        self.right_panel = None
        super().__init__(parent, title, data)

    def add_selectors(self, parent):
        labels = list(self.options)

        self.left_var.set(labels[0])
        self.right_var.set("velocity v")

        for text, variable in (("left", self.left_var), ("right", self.right_var)):
            ttk.Label(parent, text=text).pack(side="left", padx=(0, 4))
            box = ttk.Combobox(
                parent, textvariable=variable, values=labels, state="readonly", width=30
            )
            box.pack(side="left", padx=(0, 14))
            box.bind("<<ComboboxSelected>>", lambda event: self.rebuild())

    def build_panels(self, snapshot):
        ax_left = self.fig.add_subplot(1, 2, 1, projection="3d")
        ax_right = self.fig.add_subplot(1, 2, 2, projection="3d")

        self.left_panel = mx.SurfacePanel(
            self.fig, ax_left, self.grid, snapshot, self.options[self.left_var.get()]
        )
        self.right_panel = mx.SurfacePanel(
            self.fig, ax_right, self.grid, snapshot, self.options[self.right_var.get()]
        )

        self.fig.subplots_adjust(left=0.03, right=0.98, bottom=0.05, top=0.90, wspace=0.16)

    def refresh(self, snapshot):
        self.left_panel.update(snapshot)
        self.right_panel.update(snapshot)


# ---------------------------------------------------------------------------
# Sidebar widgets
# ---------------------------------------------------------------------------


class CollapsibleSection:
    """A labelled block in the sidebar that folds away when not needed."""

    def __init__(self, parent, title, expanded=False):
        self.title = title
        self.expanded = expanded

        self.outer = ttk.Frame(parent)
        self.outer.pack(fill="x", pady=5)

        self.header = ttk.Button(self.outer, text=self._header_text(), command=self.toggle)
        self.header.pack(fill="x")

        self.body = ttk.Frame(self.outer, padding=8)

        if expanded:
            self.body.pack(fill="x")

    def _header_text(self):
        arrow = ARROW_OPEN if self.expanded else ARROW_CLOSED
        return f"{arrow}  {self.title}"

    def toggle(self):
        self.expanded = not self.expanded
        self.header.configure(text=self._header_text())

        if self.expanded:
            self.body.pack(fill="x")
        else:
            self.body.forget()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Incompressible mixing")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda event: self.root.attributes("-fullscreen", False))
        self.root.bind("<F11>", self.toggle_fullscreen)

        self.output_dir = os.path.abspath("results")
        self.tabs = []

        self.mode_var = tk.StringVar(value="gradient-flow")
        self.N_var = tk.StringVar(value="128")
        self.time_var = tk.StringVar(value=str(mx.DEFAULT_FINAL_TIME["gradient-flow"]))
        self.cfl_var = tk.StringVar(value="0.45")
        self.snapshots_var = tk.StringVar(value="180")
        self.rho_var = tk.StringVar(value=mx.DEFAULT_MIXING_RHO_EXPR)
        self.lambda_var = tk.StringVar(value=mx.DEFAULT_LAMBDA_EXPR)

        self._previous_mode = self.mode_var.get()
        self.mode_var.trace_add("write", self.on_mode_change)

        self.build_layout()

    def toggle_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def on_mode_change(self, *_):
        """Swap in the other mode's defaults, unless the user changed them.

        The geodesic equations are usually run from a zero-mean rho on a short
        clock; the gradient flow wants a positive density with some structure to
        flatten out, and a much longer one.  Anything the user has typed by hand
        is left alone.
        """
        mode = self.mode_var.get()
        previous = self._previous_mode
        self._previous_mode = mode

        rho_defaults = {
            "geodesic": mx.DEFAULT_RHO_EXPR,
            "gradient-flow": mx.DEFAULT_MIXING_RHO_EXPR,
        }

        if mx.tidy_expression(self.rho_var.get()) == mx.tidy_expression(rho_defaults[previous]):
            self.rho_var.set(rho_defaults[mode])

        if self.time_var.get().strip() == str(mx.DEFAULT_FINAL_TIME[previous]):
            self.time_var.set(str(mx.DEFAULT_FINAL_TIME[mode]))

    # -- layout --------------------------------------------------------------

    def build_layout(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", font=("Segoe UI", 9))
        style.configure("Run.TButton", font=("Segoe UI", 12, "bold"), padding=10)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        sidebar = ttk.Frame(main, width=340)
        sidebar.pack(side="left", fill="y", padx=(0, 12))
        sidebar.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        panel = self.build_scrollable_sidebar(sidebar)

        header = ttk.Frame(panel)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Mixing", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="RUN", style="Run.TButton", command=self.run).pack(side="right")

        modes = ttk.LabelFrame(panel, text="Equations")
        modes.pack(fill="x", pady=6)

        for value, text in (
            ("geodesic", "geodesic equations (\u00a76.1)"),
            ("gradient-flow", "H\u207b\u00b9 gradient flow (\u00a77.1)"),
        ):
            ttk.Radiobutton(modes, text=text, variable=self.mode_var, value=value).pack(
                anchor="w", padx=10, pady=4
            )

        numerics = CollapsibleSection(panel, "Grid and time", expanded=True)
        self.add_entry(numerics.body, "grid size N", self.N_var)
        self.add_entry(numerics.body, "final time", self.time_var)
        self.add_entry(numerics.body, "CFL number", self.cfl_var)
        self.add_entry(numerics.body, "display snapshots", self.snapshots_var)

        initial = CollapsibleSection(panel, "Initial conditions", expanded=True)
        self.add_entry(initial.body, "rho(x, y)", self.rho_var, width=22)
        self.add_entry(initial.body, "lambda(x, y)", self.lambda_var, width=22)
        ttk.Label(
            initial.body,
            text="lambda is only used by the geodesic equations. "
            "Write formulas in x and y: sin(x)*cos(y), exp(cos(x)), 2 + 0.4*cos(2*x + y).",
            style="Hint.TLabel",
            wraplength=290,
            justify="left",
        ).pack(fill="x", padx=8, pady=(4, 2))

        output = CollapsibleSection(panel, "Output", expanded=False)
        ttk.Label(
            output.body, text=self.output_dir, style="Hint.TLabel", wraplength=290, justify="left"
        ).pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(output.body, text="open output folder", command=self.open_output).pack(
            fill="x", padx=8
        )

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

    def build_scrollable_sidebar(self, parent):
        """The settings column scrolls, since it does not fit on short screens."""
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)

        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas, padding=(0, 0, 8, 0))
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))

        def scroll(event):
            # Button-4/5 are the X11 wheel events; delta is used everywhere else.
            step = -1 if event.num == 4 else 1 if event.num == 5 else -event.delta // 120
            canvas.yview_scroll(int(step), "units")

        def grab_wheel(_event):
            for sequence in WHEEL_EVENTS:
                self.root.bind_all(sequence, scroll)

        def release_wheel(_event):
            for sequence in WHEEL_EVENTS:
                self.root.unbind_all(sequence)

        # Only steal the wheel while the pointer is actually over the sidebar,
        # otherwise scrolling a figure would scroll the settings column.
        canvas.bind("<Enter>", grab_wheel)
        canvas.bind("<Leave>", release_wheel)

        return inner

    def add_entry(self, parent, label, variable, width=12):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=5)
        ttk.Label(row, text=label, width=18).pack(side="left")
        ttk.Entry(row, textvariable=variable, width=width).pack(side="right")

    def open_output(self):
        os.makedirs(self.output_dir, exist_ok=True)
        open_folder(self.output_dir)

    # -- running -------------------------------------------------------------

    def read_settings(self) -> mx.Settings:
        return mx.Settings(
            N=int(self.N_var.get()),
            final_time=float(self.time_var.get()),
            cfl=float(self.cfl_var.get()),
            max_snapshots=int(self.snapshots_var.get()),
            rho_expr=self.rho_var.get(),
            lambda_expr=self.lambda_var.get(),
        ).validate()

    def clear_tabs(self):
        for tab in self.tabs:
            tab.destroy()

        self.tabs = []

        for item in self.notebook.tabs():
            self.notebook.forget(item)

    def run(self):
        try:
            settings = self.read_settings()
            mode = self.mode_var.get()

            self.root.configure(cursor="watch")
            self.root.update_idletasks()

            try:
                data = mx.SOLVERS[mode](settings)
                paths = mx.save_run_figures(data, self.output_dir)
            finally:
                self.root.configure(cursor="")

            self.clear_tabs()

            for tab in (
                OverviewTab(self.notebook, "overview", data),
                SingleQuantityTab(self.notebook, "single quantity", data),
                CompareTab(self.notebook, "compare", data),
                SimpleTab(self.notebook, "energy history", mx.figure_energy_history(data)),
                SimpleTab(self.notebook, "diagnostics", mx.figure_diagnostics(data)),
                TextTab(
                    self.notebook,
                    "numbers",
                    mx.diagnostics_report(data)
                    + "\n\nFigures written to:\n  "
                    + "\n  ".join(paths),
                ),
            ):
                self.tabs.append(tab)

        except Exception as exc:  # a bad formula or an unstable run lands here
            messagebox.showerror("Run failed", str(exc))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
