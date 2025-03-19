#!/home/satyend/.conda/envs/FTS/bin/python

import cupy as xp
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse

parser = argparse.ArgumentParser(description="Run a RDF calculation.")
parser.add_argument("--state-pkl", dest="state",  type=str, action="store", default="field.pkl", help="Enter address of pickle file to dump simulation object.")
parser.add_argument("--img",       dest="img",    type=str, action="store", default="rdf.png",   help="Enter address of image to make.")
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

if __name__=="__main__":

	# open up the fields
	fields = open(args.state, "rb")
	fts    = pickle.load(fields)
	fields.close()

	# get the density fields
	RHO_P_FIELDS = fts.RHO_P_FIELDS

	# print out some information
	print(f"Mesh fineness is {fts.mesh_fineness}.", flush=True)
	print(f"Current step is {fts.current_step}.", flush=True)
	print(f"Number of stored fields is {len(RHO_P_FIELDS)}.", flush=True)

	rg_store = []
	for rho in RHO_P_FIELDS:
		rg_store.append(calc_rg2(fts, rho))
	
	fig = plt.figure(figsize=(6,6))
	ax  = plt.axes()
	ax.plot(range(len(rg_store)), rg_store, marker='o', color='steelblue', mec='k', lw=1, ls='--', label="Rg")
	ax.legend()
	ax.set_xlabel("$\\theta$")
	ax.set_ylabel("Rg")
	fig.savefig(args.img, dpi=1200, bbox_inches="tight")

	print(f"Done!", flush=True)

