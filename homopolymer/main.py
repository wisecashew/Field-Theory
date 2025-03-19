#!/home/satyend/.conda/envs/FTS/bin/python

# Joshua Lequieu 5/7/2024

import homopolymer as hp

if __name__=="__main__":

	# set up the inputs in a dictionary format
	info = {}
	info["u0"]             = 0.007                                 # this is the contact potential
	info["T"]              = 1.0                                   # this is the temperature of the simulation
	info["L"]              = 16.0                                  # this is the length of the box
	info["Dp"]             = 64                                    # this is the degree of polymerization
	info["rho0"]           = 2                                     # this is the number density of the solvent
	info["a"]              = 0.5                                   # this is the radius of each particle
	info["b0"]             = 1.0                                   # this is the length of a polymer bond
	info["mesh_fineness"]  = 100                                   # this is the degree of discretization
	info["dt"]             = 0.001                                 # this is the size of the timestep
	info["nsteps"]         = int(1e+5)                             # this is the total number of timesteps 
	info["freq"]           = 1e+2                                  # this is the frequency of dumps
	info["Kp"]             = 3 / 2 * info["T"] / (info["b0"] ** 2) # this is the spring constant
	info["field_file"]     = "field.pkl"                           # this is the field object pickle file
	info["operators_file"] = "operators.dat"                       # this is the file where information is dumped

	# get the object to run a field theoretic simulation
	fts = hp.Homopolymer(info)
	
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
