
import pytest
import h5py
import numpy as np
import tempfile
import os

@pytest.fixture
def mock_hdf5_file():
    """Generates a temporary HDF5 file with a structure mimicking brainsets."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)

    with h5py.File(path, 'w') as f:
        # Create RegularTimeSeries group
        ts_group = f.create_group("RegularTimeSeries")
        
        # Channel 1: "Voltage" - 1000 Hz, 10 seconds
        data_v = np.random.randn(10, 10000)  # 10 channels, 10000 samples
        dset_v = ts_group.create_dataset("Voltage", data=data_v)
        dset_v.attrs["sampling_rate"] = 1000.0
        
        # Channel 2: "Current" - 500 Hz, 10 seconds
        data_c = np.random.randn(2, 5000)   # 2 channels, 5000 samples
        dset_c = ts_group.create_dataset("Current", data=data_c)
        dset_c.attrs["sampling_rate"] = 500.0

        # Create Interval group
        int_group = f.create_group("Intervals")
        
        # Interval 1: "Trials"
        # columns: [start, stop]
        trials_data = np.array([
            [0.5, 1.5],
            [2.0, 3.0],
            [5.5, 6.0]
        ])
        dset_trials = int_group.create_dataset("Trials", data=trials_data)
        dset_trials.attrs["columns"] = ["start", "stop"]
        
        # Interval 2: "Stimulation"
        stim_data = np.array([
            [1.0, 1.1],
            [2.5, 2.6],
            [5.8, 5.9]
        ])
        dset_stim = int_group.create_dataset("Stimulation", data=stim_data)
        dset_stim.attrs["columns"] = ["start", "stop"]

    yield path

    # Cleanup
    if os.path.exists(path):
        os.remove(path)
