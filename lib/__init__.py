import os


cwd = os.getcwd()
data_dir = f"{cwd}/data"
tmp_dir = f"{data_dir}/tmp"
data_file = f"{data_dir}/data.h5"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
