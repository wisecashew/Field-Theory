#!/home/satyend/.conda/envs/FTS/bin/python

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import argparse

parser = argparse.ArgumentParser(description="Run a RDF calculation.")
parser.add_argument("--dir-path",  dest="dp",    type=str, action="store", required=True, metavar="/path/to/dir/", help="Enter address of directory where all the temperature files are kept.")
parser.add_argument("--state-pkl", dest="state", type=str, action="store", required=True, metavar="field.pkl",     help="Enter address of pickle file to dump simulation object.")
parser.add_argument("--img",       dest="img",   type=str, action="store", required=True, metavar="my_rg.png",     help="Enter address of image to make.")
args = parser.parse_args()

def probe_compute(fts_file, temp_vals, pres, chem_pot, temps):
	dir_addr = os.path.dirname(fts_file)
	output_file = os.path.join(dir_addr, "field.out")
	pattern = re.compile(r"Simulation is broken")
	with open(output_file, 'r') as f:
		for line in f:
			if pattern.search(line):
				return

	op_file = os.path.join(dir_addr, "operators.dat")
	df      = pd.read_csv(op_file, sep=r'\s+', engine="python", names=["step", "beta_mu", "pressure", "rho", "logQ"], skiprows=1)
	pressure.append((np.mean(df["pressure"].values[-100:]), np.std(df["pressure"].values[-100:])))
	chem_pot.append((np.mean(df["beta_mu"].values[-100:]), np.std(df["beta_mu"].values[-100:])))
	temps.append(temp_vals)
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
	pressure = []
	mu       = []
	temps    = []

	for tv, tp in zip(temp_vals, temp_paths):
		print(f"In {tp}.", flush=True)
		probe_compute(tp, tv, pressure, mu, temps)
	
	# set up the chemical potentials
	mu       = np.array(mu)
	pressure = np.array(pressure)
	temps    = np.array(temps)

	# set up the pressure figure
	fig = plt.figure(figsize=(6,6), num=0)
	ax  = plt.axes()
	ax.set_xscale("log")
	ax.errorbar(temps, pressure[:,0], yerr=pressure[:,1], marker='o', color="steelblue", mec='k', lw=1, ls='--', label="pressure")
	ax.legend()
	ax.set_xlabel("Temperature")
	ax.set_ylabel("Pressure")
	fig.savefig(args.img+"_pres", dpi=1200, bbox_inches="tight")

	# set up the mu figure
	fig = plt.figure(figsize=(6,6), num=1)
	ax  = plt.axes()
	ax.set_xscale("log")
	ax.errorbar(temps, mu[:,0], yerr=mu[:,1], marker='o', color="coral", mec='k', lw=1, ls='--', label="chem. pot.")
	ax.legend()
	ax.set_xlabel("Temperature")
	ax.set_ylabel("Chemical Potential")
	fig.savefig(args.img+"_mu", dpi=1200, bbox_inches="tight")
	print(f"Done!", flush=True)

