#!/home/satyend/.conda/envs/FTS/bin/python

# Joshua Lequieu 5/7/2024

import monatomic_fluid as monaf
import argparse 
parser = argparse.ArgumentParser(description="Runs a monatomic fluid FTS simulation.")
parser.add_argument("--field-pkl",     dest="fp",  type=str,   action="store", default="field.pkl",     help="Enter address of pickle file to dump simulation object.")
parser.add_argument("--operator-dump", dest="op",  type=str,   action="store", default="operators.dat", help="Enter address of dump file for thermodynamic properties.")
parser.add_argument("--rho",           dest="rho", type=float, action="store", required=True,           help="Enter mean density of simulation.")
args = parser.parse_args()

if __name__=="__main__":
	# this is the place where I create the object to feed into the Monatomic_Fluid class
	info = {}
	info["u0"]             = 1                # this is the contact potential
	info["T"]              = 1.0              # this is the temperature of the simulation
	info["rho0"]           = args.rho         # this is the density of the box
	info["L"]              = 16.0             # this is the length of the box
	info["a"]              = 0.5              # this is the radius of each particle
	info["mesh_fineness"]  = 100              # this is the discretization
	info["dt"]             = 0.01             # this is the size of the timestep
	info["nsteps"]         = int(5e+5)        # this is the total number of timesteps
	info["freq"]           = int(1e+4)        # this is the frequency of dumps
	info["field_file"]     = args.fp          # this is the field object pickle file
	info["operators_file"] = args.op          # this is the file where information is dumped

	# get the object to run a field theoretic simulation
	fts = monaf.Monatomic_Fluid(info)
	
	# set up the thermodynamic and geometric parameters
	fts.initialize_thermodynamic_geometric_parameters()

	# print out the information
	fts.print_information()

	# initialize the grid
	fts.initialize_grid()
	
	# initialize the fields
	fts.initialize_fields()
	
	# run the complex langevin sampling
	fts.run_complex_langevin()
	
	# save the field instance of the field
	fts.dump_field_object()

	# end of program
