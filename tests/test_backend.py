
import pytest
import numpy as np
from src.backend import DataHandler

def test_metadata_extraction(mock_hdf5_file):
    handler = DataHandler(mock_hdf5_file)
    
    assert "Voltage" in handler.metadata["time_series"]
    assert "Current" in handler.metadata["time_series"]
    assert "Trials" in handler.metadata["intervals"]
    assert "Stimulation" in handler.metadata["intervals"]
    
    meta_v = handler.metadata["time_series"]["Voltage"]
    assert meta_v["sampling_rate"] == 1000.0
    assert meta_v["n_channels"] == 10
    assert meta_v["n_samples"] == 10000
    assert meta_v["duration"] == 10.0
    
    handler.close()

def test_get_data_slicing(mock_hdf5_file):
    handler = DataHandler(mock_hdf5_file)
    
    # Request 1 second slice from 1.0 to 2.0
    time, data = handler.get_data("Voltage", 1.0, 2.0)
    
    assert data.shape == (10, 1000) # 10 channels, 1000 samples (1s * 1000Hz)
    assert len(time) == 1000
    assert np.isclose(time[0], 1.0)
    
    # Test out of bounds
    time, data = handler.get_data("Voltage", 11.0, 12.0)
    assert len(time) == 0
    assert len(data) == 0
    
    handler.close()

def test_get_interval_filtering(mock_hdf5_file):
    handler = DataHandler(mock_hdf5_file)
    
    # Trials are at [0.5, 1.5], [2.0, 3.0], [5.5, 6.0]
    # Request slice covering the middle one
    df = handler.get_interval("Trials", 1.8, 3.2)
    
    assert len(df) >= 1
    # Should definitely include the [2.0, 3.0] trial
    assert ((df["start"] == 2.0) & (df["stop"] == 3.0)).any()
    
    handler.close()
