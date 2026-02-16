"""
BrainScope Dashboard — EEG-style montage viewer.

Architecture
============
* **Single stacked plot** — all channels vertically offset in one Bokeh figure.
* **Fixed window navigation** — 30 s default; forward/back buttons + time-jump input.
* **Min-max downsampling** so Bokeh never gets more than ~4 000 points per channel.
* **Per-channel amplitude** — global scale + individual channel multiplier + lock group.
* **Interval overlay** — coloured ``hv.VSpan`` rectangles with optional text labels.
"""

import panel as pn
import holoviews as hv
import numpy as np

from src.backend import DataHandler

hv.extension("bokeh")
pn.extension()

# ─────────────────────────────── palettes ────────────────────────────────────
# Alternating blue / red / green (like the reference EEG viewer image)
_CH_COLORS = [
    "#2166ac", "#d6604d", "#1b7837", "#762a83",
    "#2166ac", "#d6604d", "#1b7837", "#762a83",
    "#2166ac", "#d6604d", "#1b7837", "#762a83",
]

_IV_COLORS = [
    "#66ccff", "#99ff99", "#ffcc66", "#ff99cc",
    "#cc99ff", "#66ffcc",
]


def _downsample_minmax(time, data, max_points_per_ch=3000):
    """Min-max decimation: preserves peaks while cutting point count."""
    n = len(time)
    if n <= max_points_per_ch:
        return time, data

    factor = max(1, n // (max_points_per_ch // 2))
    n_win = n // factor
    trim = n_win * factor

    t_win = time[:trim].reshape(n_win, factor)
    t_mid = t_win.mean(axis=1)

    n_ch = data.shape[1]
    d_win = data[:trim].reshape(n_win, factor, n_ch)
    d_min = d_win.min(axis=1)
    d_max = d_win.max(axis=1)

    # Interleave min/max for each window
    t_out = np.repeat(t_mid, 2)
    d_out = np.empty((n_win * 2, n_ch), dtype=data.dtype)
    d_out[0::2] = d_min
    d_out[1::2] = d_max
    return t_out, d_out


class Dashboard:
    """Two-phase UI: load → explore.  Explore uses a single stacked montage."""

    def __init__(self):
        self.data_handler = None
        self._ch_scales = {}          # ch_name → float multiplier
        self._all_ch_names = []       # ordered list after loading

        # ───── file loading ──────────────────────────────────────────────
        self.file_input = pn.widgets.TextInput(
            name="HDF5 File Path", placeholder="/path/to/file.h5", width=330,
        )
        self.load_button = pn.widgets.Button(
            name="Load File", button_type="primary", width=330,
        )

        # ───── group / key selection ─────────────────────────────────────
        self.group_select = pn.widgets.Select(name="Signal Group", width=330)
        self.data_key_select = pn.widgets.Select(name="Data Key", width=330)

        # ───── channel visibility ────────────────────────────────────────
        self.channel_vis = pn.widgets.CheckBoxGroup(
            name="Visible Channels", inline=False,
        )


        # ───── intervals ─────────────────────────────────────────────────
        self.interval_select = pn.widgets.MultiSelect(
            name="Intervals", size=4, width=350,
        )
        self.interval_clear = pn.widgets.Button(
            name="Clear Selection", width=350, button_type="default",
        )
        self.interval_label_field = pn.widgets.Select(
            name="Label Field", options=["(none)"], width=350,
        )

        # ───── time navigation ───────────────────────────────────────────
        self.time_slider = pn.widgets.FloatSlider(
            name="Time Position", start=0.0, end=1.0, step=1.0, value=0.0, width=350,
        )
        self.duration_text = pn.widgets.StaticText(name="Duration", value="0.0 s", width=350)
        
        self.time_window = pn.widgets.FloatSlider(
            name="Window Size (s)", value=30.0, step=1.0, start=1.0, end=60.0, width=350,
        )
        self.nav_back = pn.widgets.Button(name="◀◀", width=70, button_type="light")
        self.nav_fwd = pn.widgets.Button(name="▶▶", width=70, button_type="light")

        # ───── amplitude / scaling ───────────────────────────────────────
        self.global_scale = pn.widgets.FloatSlider(
            name="Global Amplitude", start=0.05, end=5.0, step=0.05, value=1.0, width=350,
        )
        self.ch_scale_select = pn.widgets.Select(
            name="Channel to Scale", width=350,
        )
        self.ch_scale_slider = pn.widgets.FloatSlider(
            name="Channel Amplitude", start=0.1, end=10.0, step=0.1, value=1.0, width=350,
        )
        # Removed lock_group

        # ───── colour ───────────────────────────────────────────────────
        self.line_color = pn.widgets.ColorPicker(
            name="Line Colour Override", value="#2166ac", width=350,
        )
        self.use_color_override = pn.widgets.Checkbox(
            name="Use single colour for all", value=False,
        )

        # ───── status ───────────────────────────────────────────────────
        self.status = pn.pane.Alert("Ready", alert_type="info", width=350)

        # ───── plot area ─────────────────────────────────────────────────
        # Using a RangeX stream to perform bidirectional binding
        self.range_stream = hv.streams.RangeX()
        self.range_stream.add_subscriber(self._on_x_range_change)

        self.plot_pane = pn.pane.HoloViews(
            hv.Curve([], "Time", " ").opts(height=600, responsive=True),
            sizing_mode="stretch_both",
        )

        # ───── wire callbacks ────────────────────────────────────────────
        self.load_button.on_click(self._on_load)
        self.group_select.param.watch(self._on_group_change, "value")
        self.interval_select.param.watch(self._on_interval_change, "value")
        self.interval_clear.on_click(lambda e: self.interval_select.param.update(value=[]))  # Clear handler
        self.nav_back.on_click(lambda e: self._navigate(-1))
        self.nav_fwd.on_click(lambda e: self._navigate(+1))
        self.ch_scale_select.param.watch(self._on_ch_select_change, "value")
        self.ch_scale_slider.param.watch(self._on_ch_scale_change, "value")

        # Auto-refresh on these changes
        for w in (
            self.time_slider, self.time_window, self.global_scale,
            self.channel_vis, self.use_color_override, self.line_color,
            self.interval_select, self.interval_label_field,
        ):
            w.param.watch(self._refresh, "value")

    # =====================================================================
    # Range handling for pan/scroll
    # =====================================================================

    def _on_x_range_change(self, x_range):
        """Called when panning via mouse interaction."""
        if x_range is None:
            return
        start, end = x_range
        # Update slider without triggering _refresh to avoid loops
        # We need to guard against recursion
        if abs(self.time_slider.value - start) > 0.001:
             self.time_slider.param.update(value=start)

    # =====================================================================
    # Phase 1 — loading
    # =====================================================================

    def _on_load(self, event):
        path = self.file_input.value.strip()
        if not path:
            self._set_status("⚠️ Enter a file path.", "warning")
            return
        try:
            if self.data_handler:
                self.data_handler.close()
            self.data_handler = DataHandler(path)

            ts = self.data_handler.get_timeseries_names()
            iv = self.data_handler.get_interval_names()

            self.group_select.options = ts
            self.group_select.value = ts[0] if ts else None
            self.interval_select.options = iv
            self.interval_select.value = []
            self._set_status(f"✅ Loaded **{path}**", "success")
        except Exception as exc:
            self._set_status(f"❌ {exc}", "danger")

    def _on_group_change(self, event):
        if not self.data_handler or not self.group_select.value:
            return
        grp = self.group_select.value
        keys = self.data_handler.get_data_keys_for(grp)
        self.data_key_select.options = keys
        self.data_key_select.value = "signal" if "signal" in keys else (keys[0] if keys else None)

        ch_names = self.data_handler.get_channel_names(grp)
        self._all_ch_names = ch_names
        self._ch_scales = {n: 1.0 for n in ch_names}

        # Reset display settings
        self.time_slider.value = 0.0
        self.time_window.value = 30.0
        self.global_scale.value = 1.0
        
        # Update duration info
        meta = (
            self.data_handler.metadata["regular_time_series"].get(grp)
            or self.data_handler.metadata["irregular_time_series"].get(grp)
        )
        duration = meta["duration"] if meta else 100.0
        self.time_slider.end = max(duration - self.time_window.value, 0.1)
        self.duration_text.value = f"{duration:.2f} s"

        self.channel_vis.options = ch_names
        self.channel_vis.value = ch_names  # all visible by default

        self.ch_scale_select.options = ch_names
        self.ch_scale_select.value = ch_names[0] if ch_names else None
        # lock_group removed

        self._refresh()

    # =====================================================================
    # Per-channel scaling
    # =====================================================================

    def _on_ch_select_change(self, event):
        name = self.ch_scale_select.value
        if name and name in self._ch_scales:
            self.ch_scale_slider.value = self._ch_scales[name]

    def _on_ch_scale_change(self, event):
        name = self.ch_scale_select.value
        if not name:
            return
        new_val = self.ch_scale_slider.value
        
        # Removed lock logic
        self._ch_scales[name] = new_val

        self._refresh()

    def _on_interval_change(self, event):
        if not self.data_handler:
            return
        fields = set()
        for iv in self.interval_select.value:
            fields.update(self.data_handler.get_interval_label_fields(iv))
        opts = ["(none)"] + sorted(fields)
        self.interval_label_field.options = opts
        self.interval_label_field.value = opts[1] if len(opts) > 1 else "(none)"

    # =====================================================================
    # Navigation
    # =====================================================================

    def _navigate(self, direction):
        """Shift by half a window."""
        step = self.time_window.value * 0.5
        new_val = self.time_slider.value + direction * step
        # Clamp
        new_val = max(self.time_slider.start, min(new_val, self.time_slider.end))
        self.time_slider.value = new_val 

    # =====================================================================
    # Plot rendering
    # =====================================================================

    def _refresh(self, event=None):
        """Re-render the plot based on current widget values."""
        if not self.data_handler or not self.group_select.value:
            self.plot_pane.object = hv.Curve([], "Time", " ").opts(
                height=600, responsive=True,
            )
            return

        grp = self.group_select.value
        key = self.data_key_select.value
        ch_names_vis = self.channel_vis.value
        if not ch_names_vis:
            self.plot_pane.object = hv.Curve([], "Time", " ").opts(
                height=600, responsive=True,
            )
            return

        ch_indices = [self._all_ch_names.index(n) for n in ch_names_vis]

        t_start = self.time_slider.value
        t_end = t_start + self.time_window.value

        # Fetch data
        time, data = self.data_handler.get_data(grp, key, t_start, t_end, ch_indices)
        if len(time) == 0:
            self.plot_pane.object = hv.Curve([], "Time", " ").opts(
                height=600, responsive=True,
            )
            return

        # Downsample but keep enough points for tooltips to feel responsive
        time, data = _downsample_minmax(time, data, max_points_per_ch=5000)

        n_vis = len(ch_names_vis)
        spacing = 1.0
        curves = []
        yticks = []

        use_single = self.use_color_override.value
        single_col = self.line_color.value

        for i, ch_name in enumerate(ch_names_vis):
            d = data[:, i].copy()

            # Auto-scale: normalise to unit std
            std = np.std(d)
            mean = np.mean(d)
            if std > 0:
                d = (d - mean) / std * 0.35
            else:
                d = d - mean

            # Per-channel + global scale
            ch_s = self._ch_scales.get(ch_name, 1.0)
            d *= ch_s * self.global_scale.value

            # Vertical offset (first channel at top)
            y_off = (n_vis - 1 - i) * spacing
            d += y_off

            color = single_col if use_single else _CH_COLORS[i % len(_CH_COLORS)]

            # Use a dummy dimension for hover tooltip content if needed
            # Here we just plot Time vs Amplitude. The Amplitude is offset.
            # Ideally we'd show the *original* amplitude in the tooltip, but that requires more complex linking.
            # For now, we show the plot amplitude.
            curve = hv.Curve(
                (time, d), "Time", "Amplitude", label=ch_name,
            ).opts(color=color, line_width=0.8)
            curves.append(curve)
            yticks.append((y_off, ch_name))

        overlay = hv.Overlay(curves)

        # ───── intervals overlay ────────────────────────────────────────
        lf = self.interval_label_field.value
        if lf == "(none)":
            lf = None

        # Helper for consistent categorical coloring
        # We need a stable mapping from label -> color
        import hashlib
        def _get_color(label, palette=_IV_COLORS):
            if label is None: return palette[0]
            # Simple hash to index
            h = int(hashlib.sha256(str(label).encode('utf-8')).hexdigest(), 16)
            return palette[h % len(palette)]

        for iv_idx, iv_name in enumerate(self.interval_select.value):
            df = self.data_handler.get_interval(iv_name, t_start, t_end, lf)
            if df.empty:
                continue

            # If we have labels, colour by label. If not, fallback to set colour.
            if "label" in df.columns:
                # We can't vectorized color mapping easily with hv.Rectangles unless using a Dimension
                # But creating many Rectangles overlay objects is slow. 
                # Better: Add a 'color' column to df and use color='color' style mapping?
                # hv.Rectangles supports color-mapping a dimension if we declare it?
                # Actually, simplest robust way: iterate unique labels and overlay.
                # Since intervals usually aren't huge in number per view, detailed iteration is OK.
                # Optimization: map colors in DF, pass list of colors to opts?
                # opts(color='color_column') works if vdims include color_column.
                
                df["color"] = df["label"].apply(lambda x: _get_color(x, _IV_COLORS * 3)) # broader palette
                
                # We need to construct the Rectangles with vdims having color
                # data = (start, y0, end, y1, color)
                # But hv.Rectangles accepts (l, b, r, t, val...)
                
                rect_data = []
                for _, r in df.iterrows():
                    rect_data.append((
                        r["start"], -0.6, 
                        r["end"], (n_vis - 1) * spacing + 0.6, 
                        r["color"]
                    ))
                
                # vdims: 'color'
                rects_lay = hv.Rectangles(rect_data, vdims="color").opts(
                    color="color", alpha=0.35, line_width=0
                )
                overlay = overlay * rects_lay

                # Clamp label X-position
                def _clamp_center(row):
                    s = max(row["start"], t_start)
                    e = min(row["end"], t_end)
                    if s >= e: return (row["start"] + row["end"])/2
                    return (s + e) / 2

                xs = df.apply(_clamp_center, axis=1).tolist()
                ys = [(n_vis - 1) * spacing + 0.5] * len(xs)
                labs = df["label"].tolist()
                
                overlay = overlay * hv.Labels(
                    {"x": xs, "y": ys, "text": labs}, ["x", "y"], "text",
                ).opts(text_font_size="9pt", text_color="black", text_font_style="bold")

            else:
                # No labels -> single colour for the whole set
                iv_col = _IV_COLORS[iv_idx % len(_IV_COLORS)]
                rects = [
                    (r["start"], -0.6, r["end"], (n_vis - 1) * spacing + 0.6)
                    for _, r in df.iterrows()
                ]
                overlay = overlay * hv.Rectangles(rects).opts(
                    color=iv_col, alpha=0.25, line_width=0,
                )

        # ── final opts ──────────────────────────────────────────────────
        plot_height = max(500, n_vis * 45 + 80)
        
        # Attach the RangeX stream to the overlay so interactions update it
        self.range_stream.source = overlay
        
        # Configure WheelPanTool explicitly
        from bokeh.models import WheelPanTool
        # Note: 'xwheel_pan' is the string alias used in active_tools usually.
        # But if we create a custom instance, we might need to trust its internal name.
        # The internal name of WheelPanTool is 'wheel_pan'.
        # However, active_tools='wheel_pan' might ambiguous if we have multiple?
        # Let's try adding it to tools, and setting active_tools to the instance itself if allowed, 
        # or just 'wheel_pan' (which should pick up the only wheel pan tool).
        
        xwheel = WheelPanTool(dimension="width")
        
        overlay = overlay.opts(
            hv.opts.Curve(tools=["hover"]),
            hv.opts.Overlay(
                height=plot_height,
                responsive=True,
                yticks=yticks,
                ylabel="",
                xlabel="Time (s)",
                xlim=(t_start, t_end),
                ylim=(-0.8, (n_vis - 1) * spacing + 0.8),
                show_legend=False,
                # Setting active_tools=['wheel_pan'] should rely on the tool we added
                active_tools=["wheel_pan", "pan"],
                tools=[xwheel, "pan", "hover", "save", "reset"],
                toolbar="above",
            ),
        )
        self.plot_pane.object = overlay

    # =====================================================================
    # Helpers
    # =====================================================================

    def _set_status(self, msg, alert_type="info"):
        self.status.object = msg
        self.status.alert_type = alert_type

    # =====================================================================
    # Layout
    # =====================================================================

    def view(self):
        nav_row = pn.Row(self.nav_back, self.nav_fwd)

        sidebar = pn.Column(
            pn.pane.Markdown("## File"),
            self.file_input,
            self.load_button,
            pn.layout.Divider(),

            pn.pane.Markdown("## Signal"),
            self.group_select,
            self.data_key_select,
            pn.layout.Divider(),

            pn.pane.Markdown("## Channels"),
            self.channel_vis,
            pn.layout.Divider(),

            pn.pane.Markdown("## Intervals"),
            self.interval_select,
            self.interval_clear,  # Added button
            self.interval_label_field,
            pn.layout.Divider(),

            pn.pane.Markdown("## Navigation"),
            self.duration_text,
            self.time_slider,
            self.time_window,
            nav_row,
            pn.layout.Divider(),

            pn.pane.Markdown("## Amplitude"),
            self.global_scale,
            self.ch_scale_select,
            self.ch_scale_slider,
            pn.layout.Divider(),

            pn.pane.Markdown("## Colour"),
            self.use_color_override,
            self.line_color,
            pn.layout.Divider(),

            self.status,
            width=380,
            scroll=True,
        )

        main = pn.Column(
            pn.pane.Markdown("# BrainScope"),
            self.plot_pane,
            sizing_mode="stretch_both",
        )

        return pn.template.FastListTemplate(
            title="BrainScope",
            sidebar=[sidebar],
            main=[main],
            sidebar_width=400,  # Ensure template sidebar fits content
        ).servable()
