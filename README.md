
# BrainScope

**BrainScope** is a high-performance Python dashboard for visualizing large-scale Neuroscience HDF5 datasets. Built with [Panel](https://panel.holoviz.org/), [HoloViews](https://holoviews.org/), and [Datashader](https://datashader.org/), it enables interactive exploration of data that exceeds available RAM.

## 🚀 Key Features

*   **Lazy Loading**: Reads HDF5 data largely from disk, loading only the necessary slices for the current view.
*   **Big Data Visualization**: Uses **Datashader** to rasterize millions of data points server-side, ensuring fluid interactivity.
*   **Multi-Channel Support**: Visualize multiple continuous time series (e.g., Voltage, Current) on a synchronized timeline.
*   **Interval Overlays**: Overlay discrete events (e.g., Trials, Stimulation) on top of continuous signals.
*   **Modern Stack**: Managed with `uv` for fast, reliable dependency resolution.

## 🛠️ Installation

This project uses `uv` for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd brainscope
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

##  ▶️ Usage

1.  **Start the Dashboard**:
    ```bash
    uv run panel serve app.py --autoreload
    ```

2.  **Open in Browser**:
    Navigate to the URL shown in the terminal (usually `http://localhost:5006/app`).

3.  **Explore Data**:
    *   Enter the absolute path to your `.h5` file.
    *   Select channels and intervals to visualize.
    *   Pan and zoom to dynamically load data.

## 📂 Project Structure

*   `app.py`: Application entry point.
*   `src/backend.py`: `DataHandler` class for efficient HDF5 slicing and metadata extraction.
*   `src/dashboard.py`: Panel UI and HoloViews visualization logic.
*   `tests/`: Unit tests and `conftest.py` for generating mock HDF5 data.

## 🧪 Testing

Run the test suite to verify backend logic and UI components:

```bash
uv run pytest
```
