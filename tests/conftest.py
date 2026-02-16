"""Pytest fixtures: mock HDF5 file mimicking brainsets structure."""

import pytest
import h5py
import numpy as np
import tempfile
import os


@pytest.fixture
def mock_hdf5_file():
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)

    with h5py.File(path, "w") as f:
        # ── RegularTimeSeries: "eeg" ─────────────────────────────────────
        eeg = f.create_group("eeg")
        eeg.attrs["object"] = "RegularTimeSeries"
        eeg.attrs["sampling_rate"] = 1000.0
        eeg.create_dataset("signal", data=np.random.randn(10000, 4))  # 10 s, 4 ch
        eeg.create_dataset("train_mask", data=np.ones(10000, dtype=bool))

        # Channel names via channels/id
        ch_grp = f.create_group("channels")
        ch_grp.attrs["object"] = "ArrayDict"
        ch_grp.create_dataset("id", data=[b"F3", b"F4", b"C3", b"C4"])

        # ── IrregularTimeSeries: "body" ──────────────────────────────────
        body = f.create_group("body")
        body.attrs["object"] = "IrregularTimeSeries"
        ts = np.linspace(0, 10, 500)
        body.create_dataset("timestamps", data=ts)
        body.create_dataset("position_x", data=np.random.randn(500, 3))
        body.create_dataset("train_mask", data=np.ones(500, dtype=bool))

        # Channel names via body_ids/ids
        bids = f.create_group("body_ids")
        bids.attrs["object"] = "ArrayDict"
        bids.create_dataset("ids", data=[b"X", b"Y", b"Z"])

        # ── Interval: "trials" (with labels) ────────────────────────────
        trials = f.create_group("trials")
        trials.attrs["object"] = "Interval"
        trials.create_dataset("start", data=np.array([0.5, 2.0, 5.5]))
        trials.create_dataset("end", data=np.array([1.5, 3.0, 6.0]))
        trials.create_dataset("epoch", data=[b"Rest", b"Task", b"Rest"])

        # ── Interval: "domain" (no labels) ──────────────────────────────
        dom = f.create_group("domain")
        dom.attrs["object"] = "Interval"
        dom.create_dataset("start", data=np.array([0.0]))
        dom.create_dataset("end", data=np.array([10.0]))

        # ── Non-data group ───────────────────────────────────────────────
        bs = f.create_group("brainset")
        bs.attrs["object"] = "Data"

    yield path
    if os.path.exists(path):
        os.remove(path)
