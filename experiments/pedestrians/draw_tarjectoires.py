import dill
import os
import sys
import argparse
import pathlib
sys.path.append("../../trajectron")
from tqdm import tqdm, trange
from m2m_toolbox import viz_orig_gen, select_from_batch_input


parser = argparse.ArgumentParser()
parser.add_argument("--traj_dir", help="generated_data_directory_pkl", type=str)
args = parser.parse_args()

pkl_files = []
for file in os.listdir(args.traj_dir):
    if file.endswith(".pkl"):
        pkl_files.append(os.path.join(args.traj_dir, file))
print(f"Found {len(pkl_files)} plk files")
tlist = tqdm(pkl_files)
for x in tlist:
    with open(x, 'rb') as f:
        data = dill.load(f)
    n_samples = data["original"]["outputs"].shape[0]
    orig_i = data["original"]["inputs"]
    gen_i = data["generated"]["inputs"]
    save_dir = x[:-4]
    currentfile = x.split("/")[-1]
    print(f"Plotting file: {currentfile}")
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i in trange(n_samples):
        orig_sample = select_from_batch_input(orig_i, i)
        gen_sample = select_from_batch_input(gen_i, i)
        viz_orig_gen(orig_sample, gen_sample, None, i, save_dir=save_dir)
