from pathlib import Path

work_dir = Path(__file__).resolve().parent / ".." / ".."
plot_dir = work_dir / "plots"
resource_dir = work_dir / "resources"
data_dir = work_dir / "data"
tmp_dir = data_dir / "tmp"
h5_dir = data_dir / "h5"
postprocess_dir = data_dir / "postprocess"
for dir in [plot_dir, data_dir, postprocess_dir, tmp_dir, h5_dir]:
    dir.mkdir(parents=True, exist_ok=True)

data_file = data_dir / "data.h5"
analysis_file = data_dir / "analysis.h5"
