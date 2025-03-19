#!/home/satyend/.conda/envs/FTS/bin/python

import pickle
import itertools
import argparse
parser = argparse.ArgumentParser(description="Probe a simulation with a homopolymer in diatomic solvent.")
parser.add_argument("--state", "-s", dest="state", type=str, action="store", required=True, help="Enter address of state file.")
parser.add_argument("--field", "-f", dest="field", type=str, action="store", required=True, help="Enter address of fields file.")
args = parser.parse_args()

if __name__=="__main__":

	# instantiate a fields container
	fields = []

	# pop open the fields file
	with open(args.field, "rb") as field_file:
		field_list = pickle.load(field_file)
		fields.append(field_list)
	
	# pop open the state file
	with open(args.state, "rb") as state:
		fts = pickle.load(state)
	
	# print out some information about the state
	fts.print_info()

	# print out some information about the fields
	RHO_A_FIELDS = []
	RHO_B_FIELDS = []
	RHO_P_FIELDS = []
	for elem in fields:
		for fidx in range(len(elem)):
			if fidx == 0:
				RHO_A_FIELDS.extend(elem[0])
			elif fidx == 1:
				RHO_B_FIELDS.extend(elem[1])
			elif fidx == 2:
				RHO_P_FIELDS.extend(elem[2])
			else:
				print(f"Why is there an index {fidx}?", flush=True)
				print(f"Breaking out...", flush=True)
				exit()
	
	print(f"Number of dumped out field list is {len(RHO_A_FIELDS)}, {len(RHO_B_FIELDS)}, {len(RHO_P_FIELDS)}.", flush=True)
	print(f"Done with probe.", flush=True)
