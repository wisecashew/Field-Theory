import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse 
parser = argparse.ArgumentParser(description="Plot thermodynamic information.")
parser.add_argument("--operator-dump", dest="op",  type=str,   action="store", default="operators.dat", help="Enter address of dump file for thermodynamic properties.")
args = parser.parse_args()

if __name__=="__main__":

	# obtain the rest of the thermodynamic information
	df = pd.read_csv(args.op, sep='\s+', engine="python", names=["step", "mu_r", "mu_i", "P_r", "P_i", "rho_r", "lnq"], skiprows=1)
	steps   = df["step"].values
	mu_r    = df["mu_r"].values
	P_r     = df["P_r"].values
	rho_r   = df["rho_r"].values
	f       = np.array(df["lnq"].values, dtype=complex)
	P_calc  = mu_r.real * 2 - f.real / (16**3)
	# ------------------------------------
	sP_r    = P_r[-50:]                 # /np.std(P_r)
	sP_r    = (sP_r - np.mean(sP_r))      # /np.std(P_r)
	sP_calc = P_calc[-50:]
	sP_calc = (sP_calc.real - np.mean(sP_calc.real)) # /np.std(P_calc)).real
	# ------------------------------------
	print(f"scaled real pressure = {sP_r}")
	print(f"scaled calc'd pressure = {sP_calc}")
	fig = plt.figure(figsize=(6,6), num=-1)
	ax  = plt.axes()
	ax.plot(steps[-len(steps)//2:], sP_r,    c="steelblue", marker='^', ls='--')
	ax.plot(steps[-len(steps)//2:], sP_calc, c="coral",     marker='o', ls='--')
	ax.set_ylim(-0.1, 0.1)
	fig.savefig("pressure_check", dpi=1200, bbox_inches="tight")
	print(f"Mean real pressure is {np.mean(sP_r).real} and std deviation is {np.std(sP_r)}.", flush=True)
	print(f"Mean grand pressure is {np.mean(sP_calc)} and std deviation is {np.std(sP_calc)}.", flush=True)
	exit()

	fig = plt.figure(figsize=(6,6), num=0)
	ax  = plt.axes()
	ax.plot(steps, mu_r, marker='o', lw=1, ls='--', mec='k', c="coral", label="$\\mu$")
	ax.set_xlabel("Steps")
	ax.set_ylabel("$\\mu$")
	fig.savefig("mu.png", dpi=1200, bbox_inches="tight")
	plt.close()

	fig = plt.figure(figsize=(6,6), num=1)
	ax  = plt.axes()
	ax.plot(steps, P_r, marker='o', lw=1, ls='--', mec='k', c="lavender", label="$p$")
	ax.set_xlabel("Steps")
	ax.set_ylabel("$p$")
	fig.savefig("pressure.png", dpi=1200, bbox_inches="tight")
	plt.close()

	fig = plt.figure(figsize=(6,6), num=2)
	ax  = plt.axes()
	ax.plot(steps, rho_r, marker='o', lw=1, ls='--', mec='k', c="steelblue", label="$\\rho$")
	ax.set_xlabel("Steps")
	ax.set_ylabel("$\\rho$")
	fig.savefig("density.png", dpi=1200, bbox_inches="tight")
	plt.close()

	fig = plt.figure(figsize=(6,6), num=3)
	ax  = plt.axes()
	ax.plot(steps, f.real, marker='o', lw=1, ls='--', mec='k', c="darkred", label="$F$")
	ax.set_xlabel("Steps")
	ax.set_ylabel("$F$")
	fig.savefig("free_energy.png", dpi=1200, bbox_inches="tight")
	plt.close()

