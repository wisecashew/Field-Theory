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

def construct_rdf(fts, RHO_FIELD, suffix):

	# set up an accumulator array
	r           = xp.sqrt(fts.xx**2 + fts.yy**2 + fts.zz**2)
	bin_edges   = xp.arange(0, fts.L/2 + (fts.dr).get(), (fts.dr).get())
	bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
	r_flat      = r.ravel()
	rel_idx     = xp.where(r_flat<1e-12)[0][0]
	nrolls      = 100
	gr_store    = xp.zeros((len(RHO_FIELD)*nrolls, len(bin_centers)), dtype=complex)

	# loop through the different fields and calculate the radial distribution function
	for idx, rho in enumerate(RHO_FIELD):
		print(f"@ field index #{idx}.", flush=True)
		# get the coordinates of the particle in the center 
		# since x_range is from (-L/2 to L/2), i'd assume the 
		# particle is at (0, 0, 0)
		counts, _ = xp.histogram(r_flat, bins=bin_edges)
		counts    = counts.astype(xp.float32)
		shifts    = []
		for roll in range(nrolls):
			print(f"\t@ roll #{roll}.", flush=True)
			if roll != 0:
				while True:
					shift = (int(xp.random.randint(0, len(fts.x))), int(xp.random.randint(0, len(fts.x))), int(xp.random.randint(0, len(fts.x))))
					if shift in shifts:
						pass
					else:
						shifts.append(shift)
						break
			else:
				shift = (0, 0, 0)
				shifts.append(shift)
			rho_rolled = xp.roll(rho, shift=shift)
			rho_flat                  = rho_rolled.ravel()
			density_sum, _            = xp.histogram(r_flat, bins=bin_edges, weights=rho_flat*rho_flat[rel_idx])
			avg_density               = xp.zeros_like(density_sum)
			mask                      = counts > 0
			avg_density[mask]         = density_sum[mask] / counts[mask]
			gr_store[nrolls*idx+roll] = avg_density / fts.rho0 ** 2
	
	gr_mean = xp.mean(gr_store, axis=0).real
	gr_err  = xp.std(gr_store, axis=0)/xp.sqrt(len(gr_store)).real
	fig = plt.figure(figsize=(6,6))
	ax  = plt.axes()
	ax.errorbar(bin_centers.get(), gr_mean.get(), yerr=gr_err.get(), marker='o', ecolor='k', lw=1, ls='--', mec='k', c="darkgreen", label="RDF")
	ax.set_xlim(0, 8)
	ax.set_ylim(xp.min(gr_mean.get())*0.9, xp.max(gr_mean.get())*1.1)
	ax.tick_params(axis="y", rotation=45)
	ax.legend()
	ax.set_xlabel("r")
	ax.set_ylabel("g(r)")
	fig.savefig(args.img+suffix, dpi=1200, bbox_inches="tight")
	return

def construct_cross_rdf(fts, RHO1, RHO2, img_name):

	# set up an accumulator array
	r           = xp.sqrt(fts.xx**2 + fts.yy**2 + fts.zz**2)
	bin_edges   = xp.arange(0, fts.L/2 + (fts.dr).get(), (fts.dr).get())
	bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
	r_flat      = r.ravel()
	rel_idx     = xp.where(r_flat<1e-12)[0][0]
	nrolls      = 100
	gr_store    = xp.zeros((len(RHO1)*nrolls, len(bin_centers)), dtype=complex)


	# loop through the different fields and calculate the radial distribution function
	for idx, rho1 in enumerate(RHO1):
		print(f"@ field index #{idx}.", flush=True)
		# get the coordinates of the particle in the center 
		# since x_range is from (-L/2 to L/2), i'd assume the 
		# particle is at (0, 0, 0)
		counts, _ = xp.histogram(r_flat, bins=bin_edges)
		counts    = counts.astype(xp.float32)
		shifts    = []
		rho2_flat = RHO2[idx].ravel()
		for roll in range(nrolls):
			print(f"\t@ roll #{roll}.", flush=True)
			if roll != 0:
				while True:
					shift = (int(xp.random.randint(0, len(fts.x))), int(xp.random.randint(0, len(fts.x))), int(xp.random.randint(0, len(fts.x))))
					if shift in shifts:
						pass
					else:
						shifts.append(shift)
						break
			else:
				shift = (0, 0, 0)
				shifts.append(shift)

			rho1_rolled = xp.roll(rho1, shift=shift)
			rho1_flat                 = rho1_rolled.ravel()
			density_sum, _            = xp.histogram(r_flat, bins=bin_edges, weights=rho2_flat*rho1_flat[rel_idx])
			avg_density               = xp.zeros_like(density_sum)
			mask                      = counts > 0
			avg_density[mask]         = density_sum[mask] / counts[mask]
			gr_store[nrolls*idx+roll] = avg_density / fts.rho0 ** 2

	gr_mean = xp.mean(gr_store, axis=0).real
	gr_err  = xp.std(gr_store, axis=0)/xp.sqrt(len(gr_store)).real
	fig = plt.figure(figsize=(6,6))
	ax  = plt.axes()
	ax.errorbar(bin_centers.get(), gr_mean.get(), yerr=gr_err.get(), marker='o', ecolor='k', lw=1, ls='--', mec='k', c="darkgreen", label="RDF")
	ax.set_xlim(0, 8)
	ax.set_ylim(xp.min(gr_mean.get())*0.9, xp.max(gr_mean.get())*1.1)
	ax.tick_params(axis="y", rotation=45)
	ax.legend()
	ax.set_xlabel("r")
	ax.set_ylabel("g(r)")
	fig.savefig(img_name, dpi=1200, bbox_inches="tight")

	return

if __name__=="__main__":

	# open up the fields
	fields = open(args.state, "rb")
	fts    = pickle.load(fields)
	fields.close()

	# get the density fields
	RHO_A_FIELDS = fts.RHO_A_FIELDS
	RHO_B_FIELDS = fts.RHO_B_FIELDS
	RHO_P_FIELDS = fts.RHO_P_FIELDS

	# print out some information
	print(f"Mesh fineness is {fts.mesh_fineness}.", flush=True)
	print(f"Current step is {fts.current_step}.", flush=True)
	print(f"Number of stored fields is {len(RHO_A_FIELDS)}.", flush=True)

	# suffs = ["A", "B", "P"]
	construct_cross_rdf(fts, RHO_P_FIELDS, RHO_A_FIELDS, "rdf_AP")
	construct_cross_rdf(fts, RHO_P_FIELDS, RHO_B_FIELDS, "rdf_BP")
	construct_cross_rdf(fts, RHO_A_FIELDS, RHO_B_FIELDS, "rdf_AB")
	construct_cross_rdf(fts, RHO_B_FIELDS, RHO_A_FIELDS, "rdf_BA")
	# RHOs  = [RHO_A_FIELDS, RHO_B_FIELDS, RHO_P_FIELDS]
	# for i in range(3):
	# 	construct_rdf(fts, RHOs[i], suffs[i])
	print(f"Done!", flush=True)

