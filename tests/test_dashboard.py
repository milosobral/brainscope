
import pytest
import panel as pn
from src.dashboard import Dashboard

def test_dashboard_instantiation():
    dashboard = Dashboard()
    assert isinstance(dashboard.plot_pane, pn.Column)
    assert dashboard.channel_select.name == "Select Channels"

def test_dashboard_view():
    dashboard = Dashboard()
    view = dashboard.view()
    # view should be a Template object (FastListTemplate) which matches .servable() return
    assert isinstance(view, pn.template.FastListTemplate)

def test_dashboard_load_file(mock_hdf5_file):
    dashboard = Dashboard()
    dashboard.file_input.value = mock_hdf5_file
    dashboard._load_file(None)
    
    assert dashboard.status_text.value.startswith("Loaded")
    assert "Voltage" in dashboard.channel_select.options
    assert "Trials" in dashboard.interval_select.options
    
    # Select channels and trigger update
    dashboard.channel_select.value = ["Voltage"]
    dashboard._update_plot(None)
    
    # Check if plot pane is populated
    assert isinstance(dashboard.plot_pane.object, pn.pane.HoloViews)
