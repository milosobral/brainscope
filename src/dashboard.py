
import panel as pn
import holoviews as hv
import holoviews.operation.datashader as hd
from holoviews.streams import RangeX
import pandas as pd
import numpy as np
from src.backend import DataHandler

hv.extension('bokeh')

class Dashboard:
    def __init__(self):
        self.data_handler = None
        self.file_input = pn.widgets.TextInput(name="HDF5 File Path", placeholder="/path/to/file.h5")
        self.load_button = pn.widgets.Button(name="Load File", button_type="primary")
        self.channel_select = pn.widgets.MultiSelect(name="Select Channels", size=8)
        self.interval_select = pn.widgets.MultiSelect(name="Select Intervals", size=8)
        self.visualize_button = pn.widgets.Button(name="Visualize", button_type="success")
        
        self.plot_pane = pn.Column()
        self.status_text = pn.widgets.StaticText(name="Status", value="Ready")
        
        # Wiring callbacks
        self.load_button.on_click(self._load_file)
        self.visualize_button.on_click(self._update_plot)
        
    def _load_file(self, event):
        filepath = self.file_input.value
        if not filepath:
            self.status_text.value = "Please enter a file path."
            return
            
        try:
            self.data_handler = DataHandler(filepath)
            self.status_text.value = f"Loaded {filepath}"
            
            # Populate widgets
            ts_channels = list(self.data_handler.metadata["time_series"].keys())
            intervals = self.data_handler.metadata["intervals"]
            
            self.channel_select.options = ts_channels
            self.channel_select.value = [ts_channels[0]] if ts_channels else []
            
            self.interval_select.options = intervals
            self.interval_select.value = []
            
        except Exception as e:
            self.status_text.value = f"Error loading file: {e}"

    def _get_channel_curve(self, channel, x_range):
        if not self.data_handler:
            return hv.Curve([])
            
        if x_range is None:
            # Default to first 1 second or full duration if short
            # We need to look up total duration to be smart, but let's just pick a reasonable default
            # Or better: get the meta
            meta = self.data_handler.metadata["time_series"][channel]
            start, end = 0, min(10.0, meta["duration"])
        else:
            start, end = x_range
            
        time, data = self.data_handler.get_data(channel, start, end)
        
        if len(time) == 0:
            return hv.Curve([])
            
        return hv.Curve((time, data[0]), 'Time', channel) # Assuming 1st row of data for now if multi-channel array

    def _get_overlay_plot(self, x_range):
        if not self.data_handler:
            return hv.Overlay([])
            
        plots = []
        
        # 1. Time Series
        for channel in self.channel_select.value:
            # We use a localized function to capture the channel
            def callback(x_range):
                return self._get_channel_curve(channel, x_range)
            
            # Create DynamicMap
            dmap = hv.DynamicMap(callback, streams=[RangeX()])
            
            # Datashade it
            # rasterize=True makes it an Image, aggregating properly
            rasterized = hd.rasterize(dmap).opts(
                tools=['hover'], 
                active_tools=['wheel_zoom'],
                height=400, 
                responsive=True,
                title=f"Channel: {channel}"
            )
            plots.append(rasterized)
            
        # 2. Intervals
        for interval_name in self.interval_select.value:
            # For intervals, we can probably load all of them if they fit in memory, 
            # OR use logic to filter.
            # Let's assume we can load them for the current view, but better to load once if small.
            # However, requirement says lazy.
            
            def interval_callback(x_range):
                if x_range is None:
                    # Default
                     start, end = 0, 10.0
                else:
                    start, end = x_range
                    
                df = self.data_handler.get_interval(interval_name, start, end)
                if df.empty:
                     return hv.Rectangles([])
                
                # Create rects: (x0, y0, x1, y1)
                # We need to decide Y range. Spanning all plots? Or just a strip?
                # For overlay, create a transparent spans
                # hv.Rectangles expects (left, bottom, right, top)
                # We can't easily know y-range of curves without computing them.
                # A common trick is to use VSpans, or just arbitrary large Y if shared axis.
                # But hd.rasterize output might not cooperate with VSpan easily in all backends.
                # Let's use Rectangles with a fixed large Y for now, or 0-1 if normalized?
                # Actually, hv.VSpan allows infinite in Y.
                
                return hv.VSpan(df["start"], df["stop"]).opts(
                    color='red', alpha=0.2
                )
            
            # Intervals over dynamic map might be tricky if not careful with streams.
            # But let's try just overlaying a DynamicMap of intervals.
            interval_dmap = hv.DynamicMap(interval_callback, streams=[RangeX()])
            plots.append(interval_dmap)
            
        return hv.Overlay(plots).collate()

    def _update_plot(self, event):
        if not self.data_handler:
            return
            
        # We need a clear layout strategy.
        # If we have multiple channels, do we overlay them or stack them?
        # User requirement: "All selected channels must share a common X-axis"
        # Stacking is usually better for distinct signals.
        
        layout = hv.Layout()
        
        # Let's create a shared RangeX stream to synchronize them manually if needed, 
        # but HoloViews Layouts share axes by default.
        
        # Basic approach: create a list of elements
        
        plot_list = []
        
        # Shared X range stream
        range_stream = RangeX()
        
        for channel in self.channel_select.value:
            def ts_callback(x_range):
                return self._get_channel_curve(channel, x_range)
            
            dmap = hv.DynamicMap(ts_callback, streams=[range_stream])
            rasterized = hd.rasterize(dmap).opts(
                height=300, 
                responsive=True, 
                title=channel,
                xlabel='Time (s)', 
                ylabel='Amplitude'
            )
            
            # Add intervals to THIS plot
            interval_overlays = []
            for interval_name in self.interval_select.value:
                def interval_callback(x_range):
                    if x_range is None:
                         start, end = 0, 10.0 # Default fallback
                    else:
                        start, end = x_range
                    
                    df = self.data_handler.get_interval(interval_name, start, end)
                    if df.empty:
                        return hv.Rectangles([])
                    # Return list of VSpans?? No, VSpan takes iterables?
                    # Holoviews VSpan is usually one span? No, usage: hv.VSpan(x0, x1)
                    # For multiple, use Rectangles or multiple VSpans (inefficient).
                    # Efficient way: Spikes or custom geometry?
                    # VSpans not supported well in overlay for multiple disjoint regions in one element usually?
                    # Let's use Rectangles.
                    # We need Y bounds. Let's assume -1e9 to 1e9 or use the data range if possible.
                    # Rasterized image doesn't easily report range. 
                    # Let's stick to VSpan if it supports multiple bars, or just Rectangles with arbitrary Y.
                    # HoloViews defines VSpan as infinite height.
                    
                    # Correction: hv.VSpan typically represents ONE span.
                    # For multiple regions, we can use hv.Rectangles or overlays of VSpans (slow).
                    # Better: hv.Rectangles with y0=-inf, y1=inf ?? Bokeh handles this?
                    # Let's use a fixed tall height for now, e.g. -1000 to 1000.
                    
                    rects = []
                    for _, row in df.iterrows():
                        rects.append((row['start'], -10000, row['stop'], 10000))
                    return hv.Rectangles(rects).opts(color='orange', alpha=0.3)

                interval_dmap = hv.DynamicMap(interval_callback, streams=[range_stream])
                interval_overlays.append(interval_dmap)
            
            if interval_overlays:
                plot_part = rasterized * hv.Overlay(interval_overlays)
            else:
                plot_part = rasterized
                
            plot_list.append(plot_part)
            
        if not plot_list:
             self.plot_pane.object = pn.pane.Markdown("No channels selected.")
             return

        # Combine into a Layout (vertical stack)
        final_layout = hv.Layout(plot_list).cols(1).opts(shared_axes=True)
        
        self.plot_pane.object = pn.pane.HoloViews(final_layout)

    def view(self):
        sidebar = pn.Column(
            pn.pane.Markdown("## Configuration"),
            self.file_input,
            self.load_button,
            self.channel_select,
            self.interval_select,
            self.visualize_button,
            self.status_text,
            width=300
        )
        
        main = pn.Column(
            pn.pane.Markdown("# BrainScope Dashboard"),
            self.plot_pane,
            sizing_mode='stretch_both'
        )
        
        return pn.template.FastListTemplate(
            title="BrainScope",
            sidebar=[sidebar],
            main=[main]
        ).servable()
