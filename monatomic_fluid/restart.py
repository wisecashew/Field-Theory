#!/home/satyend/.conda/envs/FTS/bin/python

# Joshua Lequieu 5/7/2024

import monatomic_fluid as monaf
import pickle
import argparse
parser = argparse.ArgumentParser(description="Runs a monatomic fluid FTS simulation.")
parser.add_argument("--field-pkl",     dest="fp",  type=str,   action="store", default="field.pkl",     help="Enter address of pickle file to dump simulation object.")
args = parser.parse_args()

if __name__=="__main__":

	# load in the simulation object
	f = open(args.fp, "rb")
	fts = pickle.load(f)
	f.close()

	# print out the information
	fts.print_information()

	# run complex langevin sampling
	fts.run_complex_langevin()
	
	# save the field instance of the field
	fts.dump_field_object()

	# end of program
