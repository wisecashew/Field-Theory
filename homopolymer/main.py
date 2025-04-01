#!/home/satyend/.conda/envs/FTS/bin/python

# Joshua Lequieu 5/7/2024

import homopolymer as hp
import argparse
parser = argparse.ArgumentParser(description="Run a homopolymer.")
parser.add_argument("--input-file", "-i", dest="inp", type=str, action="store", required=True, help="Enter address of input file.")
args = parser.parse_args()

if __name__=="__main__":

	# get the object to run a field theoretic simulation
	fts = hp.Homopolymer(args.inp)
	
	# set up the thermodynamic and geometric parameters
	fts.initialize_thermodynamic_geometric_parameters()
	
	# initialize the grid
	fts.initialize_grid()
	
	# initialize the fields
	fts.initialize_fields()
	
	# run the complex langevin sampling
	fts.run_complex_langevin()
	
	# save the field instance of the field
	fts.dump_field_object()

	# end of program
