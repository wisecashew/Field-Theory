#!/home/satyend/.conda/envs/FTS/bin/python

import os
import re
import traceback
import cupy as xp
import numpy as np
import pickle
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import argparse

parser = argparse.ArgumentParser(description="Run a RDF calculation.")
parser.add_argument("--dir-path",  dest="dp",    type=str, action="store", required=True, metavar="/path/to/dir/", help="Enter address of directory where all the temperature files are kept.")
parser.add_argument("--state-pkl", dest="state", type=str, action="store", required=True, metavar="field.pkl",     help="Enter address of pickle file to dump simulation object.")
args = parser.parse_args()

# define the colormap
# set up the colors
colors = [
	(0.0, 0.0, 1.0, 1.0),  # Blue, opaque
	(1.0, 1.0, 1.0, 0.0),  # Transparent, no color (white here just for anchor)
	(1.0, 0.0, 0.0, 1.0)   # Red, opaque
]

positions = [0, 0.5, 1]

def extract_step_numbers(directory):
	step_numbers = []
	pattern = re.compile(r"step_(\d+)\.png")

	for filename in os.listdir(directory):
		match = pattern.match(filename)
		if match:
			step_numbers.append(int(match.group(1)))

	return step_numbers

def field_images(fts_file, fidx):
	# check the files
	dir_addr = os.path.dirname(fts_file)
	output_file = os.path.join(dir_addr, "field.out")
	pattern = re.compile(r"Simulation is broken")
	with open(output_file, 'r') as f:
		for line in f:
			if pattern.search(line):
				raise RuntimeError("This simulation is broken. Move on.")
	
	# Load data
	with open(fts_file, "rb") as f:
		fts = pickle.load(f)

	# get the coordinates
	x_coords = fts.xx.get().ravel()
	y_coords = fts.yy.get().ravel()
	z_coords = fts.zz.get().ravel()

	# loop over the density fields

	# set up the images
	fig    = plt.figure(figsize=(4,4))
	ax     = fig.add_subplot(111, projection="3d")
	density = fts.RHO_FIELDS[-1].real.get().ravel()

	# Create the colormap
	my_cmap = LinearSegmentedColormap.from_list("blue_x_red", list(zip(positions, colors)))

	print(f"Minimum density = {xp.min(density)}.\nMaximum density = {xp.max(density)}.", flush=True)
	sc = ax.scatter(x_coords, y_coords, z_coords, c=(density-density.min())/(density.max()-density.min()), marker='o', cmap=my_cmap)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.set_xticks([-8, 0, 8])
	ax.set_yticks([-8, 0, 8])
	ax.set_zticks([-8, 0, 8])
	ax.set_xlim3d(left=-8, right=8)
	ax.set_ylim3d(bottom=-8, top=8)
	ax.set_zbound(lower=-8, upper=8)
	fig.colorbar(sc, ax=ax, orientation="horizontal")
	img_addr = os.path.join(os.path.dirname(fts_file), "final.png")
	fig.savefig(img_addr, dpi=1200)
	plt.close()

	return

def get_sorted_temperature_folders(directory, state_pkl):

	pattern = re.compile(r"^TEMPERATURE_([-+]?\d*\.\d+|\d+)$")
	folders = []
	for folder in os.listdir(directory):
		full_path = os.path.join(directory, folder)
		if os.path.isdir(full_path):
			match = pattern.match(folder)
			if match:
				temp_value = float(match.group(1))
				folders.append((temp_value, full_path))
	
	# sort folders by the extracted float value
	folders.sort()
	temp_paths = [os.path.join(folder, state_pkl) for _, folder in folders]
	temp_vals  = [float(temp) for temp, _ in folders]

	return temp_vals, temp_paths

if __name__=="__main__":

	# get all the files you need to run `probe_compute`
	dir_path = args.dp
	pkl_file = args.state
	temp_vals, temp_paths = get_sorted_temperature_folders(dir_path, pkl_file)

	for idx, (val,tp) in enumerate(zip(temp_vals,temp_paths)):
		try:
			print(f"Processing {tp}.", flush=True)
			field_images(tp, idx)
			print(f"done!", flush=True)
		except:
			traceback.print_exc()
			print(f"Issue with file in {tp}.", flush=True)

	print(f"Done!", flush=True)

