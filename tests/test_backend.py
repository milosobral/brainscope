"""Tests for the DataHandler backend."""

import numpy as np
import pytest
from src.backend import DataHandler


class TestMetadataExtraction:
    def test_regular_ts(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        assert "eeg" in h.metadata["regular_time_series"]
        m = h.metadata["regular_time_series"]["eeg"]
        assert m["sampling_rate"] == 1000.0
        assert m["n_samples"] == 10000
        assert m["n_channels"] == 4
        assert m["duration"] == 10.0
        assert "signal" in m["data_keys"]
        h.close()

    def test_irregular_ts(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        assert "body" in h.metadata["irregular_time_series"]
        m = h.metadata["irregular_time_series"]["body"]
        assert m["n_samples"] == 500
        assert m["n_channels"] == 3
        assert "position_x" in m["data_keys"]
        assert "train_mask" not in m["data_keys"]
        h.close()

    def test_intervals(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        assert "trials" in h.metadata["intervals"]
        assert h.metadata["intervals"]["trials"]["n_intervals"] == 3
        assert "epoch" in h.metadata["intervals"]["trials"]["label_fields"]
        h.close()

    def test_ignores_non_ts(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        for cat in ("regular_time_series", "irregular_time_series", "intervals"):
            assert "brainset" not in h.metadata[cat]
        h.close()


class TestChannelNames:
    def test_regular_from_channels_id(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        names = h.get_channel_names("eeg")
        assert names == ["F3", "F4", "C3", "C4"]
        h.close()

    def test_irregular_from_group_ids(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        names = h.get_channel_names("body")
        assert names == ["X", "Y", "Z"]
        h.close()


class TestAccessors:
    def test_ts_names(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        assert set(h.get_timeseries_names()) >= {"eeg", "body"}
        h.close()

    def test_interval_names(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        assert "trials" in h.get_interval_names()
        h.close()

    def test_data_keys(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        assert "signal" in h.get_data_keys_for("eeg")
        h.close()

    def test_interval_label_fields(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        assert "epoch" in h.get_interval_label_fields("trials")
        assert h.get_interval_label_fields("domain") == []
        h.close()


class TestRegularSlicing:
    def test_shape(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        t, d = h.get_regular_data("eeg", "signal", 1.0, 2.0)
        # 1 s at 1 kHz, default → all 4 channels
        assert d.shape == (1000, 4)
        assert len(t) == 1000
        assert np.isclose(t[0], 1.0)
        h.close()

    def test_single_channel(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        t, d = h.get_regular_data("eeg", "signal", 0, 0.5, channel_indices=[2])
        assert d.shape == (500, 1)
        h.close()

    def test_out_of_bounds(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        t, d = h.get_regular_data("eeg", "signal", 11.0, 12.0)
        assert len(t) == 0
        h.close()

    def test_dispatcher(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        t, d = h.get_data("eeg", "signal", 0, 0.5, channel_indices=[0])
        assert d.shape == (500, 1)
        h.close()


class TestIrregularSlicing:
    def test_slice(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        t, d = h.get_irregular_data("body", "position_x", 2.0, 4.0)
        assert len(t) > 0
        assert d.shape[1] == 3  # all 3 channels returned by default
        assert t[0] >= 2.0 and t[-1] < 4.0
        h.close()


class TestIntervalSlicing:
    def test_filter(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        df = h.get_interval("trials", 1.8, 3.2)
        assert len(df) >= 1
        assert ((df["start"] == 2.0) & (df["end"] == 3.0)).any()
        h.close()

    def test_with_labels(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        df = h.get_interval("trials", 0, 10, label_field="epoch")
        assert "label" in df.columns
        assert "Task" in df["label"].values
        h.close()

    def test_empty(self, mock_hdf5_file):
        h = DataHandler(mock_hdf5_file)
        assert h.get_interval("trials", 100, 200).empty
        h.close()
