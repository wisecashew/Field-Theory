#!/home/satyend/.conda/envs/FTS/bin/python

import os
import re
import cupy as xp
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

def calc_rg2(fts, rho_P):
	net_mass = xp.sum(rho_P) * fts.d3r
	xcm      = xp.sum(rho_P * fts.xx) * fts.d3r / net_mass
	ycm      = xp.sum(rho_P * fts.yy) * fts.d3r / net_mass
	zcm      = xp.sum(rho_P * fts.zz) * fts.d3r / net_mass
	Rcm      = xp.array([xcm, ycm, zcm]).real
	dx       = fts.xx - Rcm[0]
	dy       = fts.yy - Rcm[1]
	dz       = fts.zz - Rcm[2]
	rg2      = xp.sum(rho_P * (dx * dx + dy * dy + dz * dz)) * fts.d3r / net_mass
	rg       = xp.sqrt(rg2)
	return rg.get()

def probe_compute(fts_file):

	# open up the fields
	fields = open(fts_file, "rb")
	fts    = pickle.load(fields)
	fields.close()

	# get the density fields
	RHO_FIELDS = fts.RHO_P_FIELDS

	# print out some information
	print(f"\tMesh fineness is {fts.mesh_fineness}.", flush=True)
	print(f"\tCurrent step is {fts.current_step}.", flush=True)
	print(f"\tNumber of stored fields is {len(RHO_FIELDS)}.", flush=True)

	rg_store = []
	for rho in RHO_FIELDS:
		rg_store.append(calc_rg2(fts, rho))
	rg_mean = xp.mean(xp.array(rg_store)[-50:]).get()
	rg_err  = xp.std(xp.array(rg_store)[-50:]).get()/xp.sqrt(10)
	return rg_mean, rg_err

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
	my_rg_mean = []
	my_rg_err  = []

	for tp in temp_paths:
		print(f"In {tp}.", flush=True)
		rg_mean, rg_err = probe_compute(tp)
		my_rg_mean.append(rg_mean)
		my_rg_err.append(rg_err)

	fig = plt.figure(figsize=(6,6))
	ax  = plt.axes()
	ax.set_xscale("log")
	ax.errorbar(temp_vals, my_rg, yerr=my_rg_err, marker='o', ec='k', color='steelblue', mec='k', lw=1, ls='--', label="Rg")
	ax.legend()
	ax.set_xlabel("Temperature")
	ax.set_ylabel("Rg")
	fig.savefig(args.img, dpi=1200, bbox_inches="tight")
	print(f"Done!", flush=True)

