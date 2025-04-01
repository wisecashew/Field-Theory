#!/home/satyend/.conda/envs/FTS/bin/python

import hp_ms
import argparse
parser = argparse.ArgumentParser(description="Runs a simulation with a homopolymer in diatomic solvent.")
parser.add_argument("--input-file", "-i", dest="inp", type=str, action="store", required=True, help="Enter address of input file.")
args = parser.parse_args()


if __name__=="__main__":

	# set up the system dictionary
	my_input_file = args.inp

	# get the field theory object
	fts = hp_ms.Homopolymer_MonatomicSolvent(my_input_file)

	# set up the thermodynamics of the simulation
	fts.initialize_thermodynamic_geometric_parameters()

	# set up the grid in which to run the simulation
	fts.initialize_grid()

	# set up the fields to facilitate the simulation
	fts.initialize_fields()

	# print out the info...
	fts.print_info()

	# blast the simulation!
	fts.run_complex_langevin()

	# save the simulation state
	fts.dump_state_object()

	# end of program
