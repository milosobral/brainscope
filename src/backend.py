
import h5py
import numpy as np
import pandas as pd

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.metadata = {
            "time_series": {},
            "intervals": []
        }
        self._open_file()
        self._extract_metadata()

    def _open_file(self):
        """Opens the HDF5 file in read mode."""
        try:
            self.file = h5py.File(self.filepath, 'r')
        except Exception as e:
            raise IOError(f"Could not open file {self.filepath}: {e}")

    def _extract_metadata(self):
        """Scans the file for TimeSeries and Interval groups."""
        if "RegularTimeSeries" in self.file:
            group = self.file["RegularTimeSeries"]
            for name in group:
                dset = group[name]
                sr = dset.attrs.get("sampling_rate", 1.0)
                shape = dset.shape
                # Assuming shape is (n_channels, n_samples)
                duration = shape[1] / sr
                self.metadata["time_series"][name] = {
                    "sampling_rate": sr,
                    "n_channels": shape[0],
                    "n_samples": shape[1],
                    "duration": duration
                }

        if "Intervals" in self.file:
            group = self.file["Intervals"]
            for name in group:
                self.metadata["intervals"].append(name)

    def get_data(self, channel_name, start_time, end_time):
        """
        Lazily loads a slice of data for the given channel.
        
        Args:
            channel_name (str): Name of the dataset in RegularTimeSeries.
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            
        Returns:
            tuple: (time_array, data_array)
        """
        if channel_name not in self.metadata["time_series"]:
            raise ValueError(f"Channel {channel_name} not found.")

        meta = self.metadata["time_series"][channel_name]
        sr = meta["sampling_rate"]
        
        # Calculate indices
        start_idx = int(max(0, start_time * sr))
        end_idx = int(min(meta["n_samples"], end_time * sr))
        
        if start_idx >= end_idx:
            return np.array([]), np.array([])
            
        # Read from disk
        dset = self.file["RegularTimeSeries"][channel_name]
        data = dset[:, start_idx:end_idx]
        
        # Generate time array
        t_start = start_idx / sr
        t_end = end_idx / sr # Exclusive
        time = np.linspace(t_start, t_end, data.shape[1], endpoint=False)
        
        return time, data

    def get_interval(self, interval_name, start_time, end_time):
        """
        Returns interval data filtered by time range.
        
        Args:
            interval_name (str): Name of the interval dataset.
            start_time (float): Start time.
            end_time (float): End time.
            
        Returns:
            pd.DataFrame: DataFrame with 'start' and 'stop' columns.
        """
        if interval_name not in self.metadata["intervals"]:
            raise ValueError(f"Interval {interval_name} not found.")
            
        dset = self.file["Intervals"][interval_name]
        # Read all for now, assuming intervals are not valid memory breakers like full TS
        # Optimization: chunked read if needed, but usually intervals are sparse.
        data = dset[:] 
        
        df = pd.DataFrame(data, columns=["start", "stop"])
        
        # Filter
        mask = (df["stop"] >= start_time) & (df["start"] <= end_time)
        return df[mask]

    def close(self):
        if self.file:
            self.file.close()

    def __del__(self):
        self.close()
