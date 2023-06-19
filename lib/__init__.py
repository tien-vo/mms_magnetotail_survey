import os


cwd = os.getcwd()
data_dir = f"{cwd}/data"
h5_dir = f"{data_dir}/h5"
tmp_dir = f"{data_dir}/tmp"
postprocess_dir = f"{data_dir}/postprocess"
for dir in [data_dir, postprocess_dir, tmp_dir, h5_dir]:
    os.makedirs(dir, exist_ok=True)

data_file = f"{data_dir}/data.h5"
