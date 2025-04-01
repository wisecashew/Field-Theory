#!/home/satyend/.conda/envs/FTS/bin/python

import hp_ms
import pickle
import argparse
parser = argparse.ArgumentParser(description="Restart a simulation with a homopolymer in monatomic solvent.")
parser.add_argument("--input-file", "-i", dest="inp", type=str, action="store", required=True, help="Enter address of input file.")
args = parser.parse_args()


if __name__=="__main__":

	# get the state of the simulation
	with open(args.inp, "rb") as f:
		fts = pickle.load(f)

	# blast the simulation!
	fts.run_complex_langevin()

	# save the simulation state
	fts.dump_state_object()

	# end of program
