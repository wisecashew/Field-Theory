#!/home/satyend/.conda/envs/FTS/bin/python

# Joshua Lequieu 5/7/2024

import pickle
import argparse
parser = argparse.ArgumentParser(description="Restart the homopolymer simulation.")
parser.add_argument("--input-file", "-i", dest="inp", type=str, action="store", required=True, help="Enter address of input file.")
args = parser.parse_args()

if __name__=="__main__":

	# get the simulation object
	with open(args.inp, 'rb') as f:
		fts = pickle.load(f)

	# run the complex langevin sampling
	fts.run_complex_langevin()
	
	# save the field instance of the field
	fts.dump_field_object()

	# end of program
