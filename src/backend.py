"""
Backend DataHandler for brainsets HDF5 files.

The brainsets/temporaldata HDF5 structure uses top-level groups named by modality
(e.g. 'eeg', 'body', 'eye', 'trials'). Each group has an '@object' attribute:
  - "RegularTimeSeries"  : @sampling_rate on group, data in datasets (n_samples, n_channels)
  - "IrregularTimeSeries": 'timestamps' dataset, data across multiple datasets
  - "Interval"           : 'start' and 'end' datasets, optional label fields
  - "ArrayDict"          : e.g. channel name look-ups
"""

import h5py
import numpy as np
import pandas as pd


class DataHandler:
    """Lazy-loading handler for brainsets HDF5 files."""

    # Known non-data datasets inside groups
    _SKIP_KEYS = frozenset({
        "timestamps", "timestamp_indices_1s",
        "train_mask", "test_mask", "valid_mask",
        "start", "end", "epoch", "timestamp",
    })

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.metadata = {
            "regular_time_series": {},
            "irregular_time_series": {},
            "intervals": {},
        }
        self._channel_name_cache = {}
        self._open_file()
        self._extract_metadata()

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _open_file(self):
        try:
            self.file = h5py.File(self.filepath, "r")
        except Exception as e:
            raise IOError(f"Could not open file {self.filepath}: {e}")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_metadata(self):
        """Walk top-level groups and classify by @object attribute."""
        for name in self.file:
            item = self.file[name]
            if not isinstance(item, h5py.Group):
                continue

            obj_type = self._attr_str(item, "object")

            if obj_type == "RegularTimeSeries":
                self._extract_regular_ts(name, item)
            elif obj_type == "IrregularTimeSeries":
                self._extract_irregular_ts(name, item)
            elif obj_type == "Interval":
                self._extract_interval(name, item)

    def _extract_regular_ts(self, name, group):
        sr = float(group.attrs.get("sampling_rate", 1.0))
        data_keys = self._get_data_keys(group)

        n_samples = 0
        n_channels = 0
        if data_keys:
            first = group[data_keys[0]]
            n_samples = first.shape[0]
            n_channels = first.shape[1] if first.ndim > 1 else 1

        self.metadata["regular_time_series"][name] = {
            "sampling_rate": sr,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "duration": n_samples / sr if sr > 0 else 0.0,
            "data_keys": data_keys,
        }

    def _extract_irregular_ts(self, name, group):
        data_keys = self._get_data_keys(group)
        n_samples = group["timestamps"].shape[0] if "timestamps" in group else 0

        ts = group["timestamps"]
        duration = float(ts[-1] - ts[0]) if n_samples > 1 else 0.0

        n_channels = 0
        if data_keys:
            first = group[data_keys[0]]
            n_channels = first.shape[1] if first.ndim > 1 else 1

        self.metadata["irregular_time_series"][name] = {
            "n_samples": n_samples,
            "n_channels": n_channels,
            "duration": duration,
            "data_keys": data_keys,
        }

    def _extract_interval(self, name, group):
        n_intervals = group["start"].shape[0] if "start" in group else 0

        # Discover label fields (string/bytes datasets that aren't start/end)
        label_fields = []
        for k in group:
            if k in ("start", "end", "domain"):
                continue
            ds = group[k]
            if isinstance(ds, h5py.Dataset) and ds.dtype.kind in ("S", "U", "O"):
                label_fields.append(k)

        self.metadata["intervals"][name] = {
            "n_intervals": n_intervals,
            "label_fields": label_fields,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attr_str(group, attr_name):
        """Read a group attribute and decode bytes to str."""
        val = group.attrs.get(attr_name, None)
        if val is None:
            return ""
        return val.decode("utf-8") if isinstance(val, bytes) else str(val)

    def _get_data_keys(self, group):
        """Return plottable dataset names in a group."""
        keys = []
        for k in group:
            if k in self._SKIP_KEYS:
                continue
            item = group[k]
            if isinstance(item, h5py.Dataset) and item.dtype.kind == "f":
                keys.append(k)
        return sorted(keys)

    # ------------------------------------------------------------------
    # Channel name look-up
    # ------------------------------------------------------------------

    def get_channel_names(self, group_name):
        """
        Return channel names for a time-series group.

        Look-up order:
          1. ``channels/id``       (common in sleep / EEG-only files)
          2. ``{group_name}_ids/ids``  (Neuro-Galaxy convention)
          3. Fall back to numeric indices.
        """
        if group_name in self._channel_name_cache:
            return self._channel_name_cache[group_name]

        meta = (
            self.metadata["regular_time_series"].get(group_name)
            or self.metadata["irregular_time_series"].get(group_name)
        )
        n_ch = meta["n_channels"] if meta else 0

        names = None

        # Strategy 1: channels/id  (one shared array)
        if "channels" in self.file:
            ch_group = self.file["channels"]
            if isinstance(ch_group, h5py.Group) and "id" in ch_group:
                ids = ch_group["id"][:]
                if len(ids) == n_ch:
                    names = [v.decode() if isinstance(v, bytes) else str(v) for v in ids]

        # Strategy 2: {group}_ids/ids
        if names is None:
            ids_key = f"{group_name}_ids"
            if ids_key in self.file:
                ids_group = self.file[ids_key]
                if isinstance(ids_group, h5py.Group) and "ids" in ids_group:
                    ids = ids_group["ids"][:]
                    if len(ids) == n_ch:
                        names = [v.decode() if isinstance(v, bytes) else str(v) for v in ids]

        if names is None:
            names = [f"ch_{i}" for i in range(n_ch)]

        self._channel_name_cache[group_name] = names
        return names

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_timeseries_names(self):
        names = list(self.metadata["regular_time_series"]) + list(self.metadata["irregular_time_series"])
        return sorted(names)

    def get_interval_names(self):
        return sorted(self.metadata["intervals"].keys())

    def get_data_keys_for(self, group_name):
        if group_name in self.metadata["regular_time_series"]:
            return self.metadata["regular_time_series"][group_name]["data_keys"]
        if group_name in self.metadata["irregular_time_series"]:
            return self.metadata["irregular_time_series"][group_name]["data_keys"]
        return []

    def get_interval_label_fields(self, interval_name):
        """Return list of available label fields for an interval."""
        meta = self.metadata["intervals"].get(interval_name)
        return meta["label_fields"] if meta else []

    # ------------------------------------------------------------------
    # Data slicing — REGULAR
    # ------------------------------------------------------------------

    def get_regular_data(self, group_name, data_key, start_time, end_time,
                         channel_indices=None):
        """
        Lazily load a time slice from a RegularTimeSeries dataset.

        Args:
            channel_indices: list of int column indices to read, or None for all.

        Returns:
            (time_1d, data_2d)  data_2d shape = (n_samples, n_channels_requested)
        """
        meta = self.metadata["regular_time_series"].get(group_name)
        if meta is None:
            raise ValueError(f"'{group_name}' is not a RegularTimeSeries.")

        sr = meta["sampling_rate"]
        n_samples = meta["n_samples"]

        s = int(max(0, start_time * sr))
        e = int(min(n_samples, end_time * sr))

        if s >= e:
            return np.array([]), np.array([]).reshape(0, 0)

        dset = self.file[group_name][data_key]
        if channel_indices is not None:
            data = dset[s:e, channel_indices]
        else:
            data = dset[s:e, :]

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        time = np.linspace(s / sr, e / sr, data.shape[0], endpoint=False)
        return time, data

    # ------------------------------------------------------------------
    # Data slicing — IRREGULAR
    # ------------------------------------------------------------------

    def get_irregular_data(self, group_name, data_key, start_time, end_time,
                           channel_indices=None):
        meta = self.metadata["irregular_time_series"].get(group_name)
        if meta is None:
            raise ValueError(f"'{group_name}' is not an IrregularTimeSeries.")

        ts_all = self.file[group_name]["timestamps"][:]
        mask = (ts_all >= start_time) & (ts_all < end_time)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return np.array([]), np.array([]).reshape(0, 0)

        s, e = int(indices[0]), int(indices[-1]) + 1
        dset = self.file[group_name][data_key]

        if dset.ndim == 1:
            data = dset[s:e].reshape(-1, 1)
        elif channel_indices is not None:
            data = dset[s:e, channel_indices]
        else:
            data = dset[s:e, :]

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return ts_all[s:e], data

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def get_data(self, group_name, data_key, start_time, end_time,
                 channel_indices=None):
        if group_name in self.metadata["regular_time_series"]:
            return self.get_regular_data(group_name, data_key, start_time, end_time, channel_indices)
        elif group_name in self.metadata["irregular_time_series"]:
            return self.get_irregular_data(group_name, data_key, start_time, end_time, channel_indices)
        else:
            raise ValueError(f"'{group_name}' not found in time series metadata.")

    # ------------------------------------------------------------------
    # Interval slicing
    # ------------------------------------------------------------------

    def get_interval(self, interval_name, start_time, end_time, label_field=None):
        """
        Return interval data filtered to time range.

        Returns:
            pd.DataFrame with columns 'start', 'end', and optionally 'label'.
        """
        if interval_name not in self.metadata["intervals"]:
            raise ValueError(f"Interval '{interval_name}' not found.")

        grp = self.file[interval_name]
        starts = grp["start"][:]
        ends = grp["end"][:]

        df = pd.DataFrame({"start": starts, "end": ends})

        if label_field and label_field in grp:
            raw = grp[label_field][:]
            df["label"] = [v.decode() if isinstance(v, bytes) else str(v) for v in raw]

        mask = (df["end"] >= start_time) & (df["start"] <= end_time)
        return df[mask].reset_index(drop=True)
