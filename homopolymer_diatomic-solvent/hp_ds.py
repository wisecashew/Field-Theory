#!/home/satyend/.conda/envs/FTS/bin/python

import cupy as xp
import re
import pickle 

class Homopolymer_DiatomicSolvent:
	# The polymer is made of monomers of species P
	# the diatomic solvent is a dumbell made of species A and B
	def __init__(self, input_file):
		self.read_input_file(input_file)
		self.current_step   = 0   # this is the current step of the complex Langevin simulation
		self.RHO_A_FIELDS   = []  # this is the set of A fields I will analyze
		self.RHO_B_FIELDS   = []  # this is the set of B fields I will analyze
		self.RHO_P_FIELDS   = []  # this is the set of P fields I will analyze
		return
	
	def read_input_file(self, input_file):
		keys = ["u0", "chi_AP", "chi_BP", "chi_AB", 'T', "Kp", "Ks", "u0",\
		  "np", "ns", "dp", 'L', 'a', "mesh_fineness", "dt",   \
		  "nsteps", "output_freq", "field_file", "state_file", "operators_file"]
		pattern = r'=\s*(.+)'
		system = dict()
		with open(input_file, 'r') as file:
			for line in file:
				tag = False
				for key in keys:
					if re.match(r"^"+key+r"\s+", line):
						tag = True
						hit = re.search(pattern, line)
						system[key] = hit.group(1)
						break
				if not tag:
					print(f"The line \"{line}\" holds weird information. Please check. Exiting...", flush=True)
					exit()
		
		# set up the parameters
		self.u0             = float(system["u0"])              # this is the strength of the excluded volume repulsion
		self.chi_AP         = float(system["chi_AP"])          # this is the cross interaction between species A and P
		self.chi_BP         = float(system["chi_BP"])          # this is the cross interaction between species B and P
		self.chi_AB         = float(system["chi_AB"])          # this is the cross interaction between species A and B
		self.T              = float(system["T"])               # this is the tempreature of the simulation
		self.Kp             = float(system["Kp"])              # this is the spring constant connecting the polymer beads
		self.Ks             = float(system["Ks"])              # this is the spring constant connecting the solvent beads
		self.np             = int(system["np"])                # this is the number of polymer molecules 
		self.ns             = int(system["ns"])                # this is the number of solvent molecules
		self.dp             = int(system["dp"])                # this is the degree of polymerization
		self.L              = float(system["L"])               # this is the length of the simulation cell
		self.a              = float(system["a"])               # this is the radius of an atom
		self.mesh_fineness  = int(system["mesh_fineness"])     # this is the fineness of the grid in units of the atomic size
		self.dt             = float(system["dt"])              # this is the size of the timestep
		self.nsteps         = int(system["nsteps"])            # this is the number of timesteps to perform
		self.freq           = int(system["output_freq"])       # this is the frequency of output
		self.state_file     = system["state_file"].strip()     # this is the state of the Hompolymer_DiatomicSolvent object
		self.field_file     = system["field_file"].strip()     # this is where I will dump the field file
		self.operators_file = system["operators_file"].strip() # this is where I will dump the operators
		return

	def SL(self, eigenval):
		sl = 1 if eigenval < 0 else 1j
		return sl

	def initialize_thermodynamic_geometric_parameters(self):
		self.us   = self.u0 / (8 * xp.pi**(3/2) * self.a**3)
		self.beta = 1 / self.T
		self.V    = self.L ** 3
		self.n    = self.dp * self.np + 2 * self.ns
		self.rho0 = (self.np * self.dp + self.ns * 2) / self.V
		self.ngrid_dim = (self.mesh_fineness, self.mesh_fineness, self.mesh_fineness)
		self.ngrid_h   = (self.ngrid_dim[0]//2, self.ngrid_dim[1]//2, self.ngrid_dim[2]//2)
		self.tot_grid  = self.ngrid_dim[0] * self.ngrid_dim[1] * self.ngrid_dim[2]
		self.d3r       = self.V / self.tot_grid
		self.INTERACTION_MATRIX = xp.array(
							[[self.u0,               self.u0 + self.chi_AB, self.u0 + self.chi_AP],
							[ self.u0 + self.chi_AB, self.u0,               self.u0 + self.chi_BP],
							[ self.u0 + self.chi_AP, self.u0 + self.chi_BP, self.u0]])
		self.Eval, self.Evec = xp.linalg.eigh(self.INTERACTION_MATRIX)
		print(f"Eigenvalues are \n{self.Eval}", flush=True)
		print(f"Eigenvectors are \n{self.Evec}", flush=True)
		return
	
	def initialize_grid(self):
		# this function initializes the grid over which we will evolve our fields
		self.x                    = xp.arange(-self.L/2, self.L/2, self.L/self.ngrid_dim[0])
		self.dr                   = xp.asarray(self.x[1] - self.x[0])
		self.xx, self.yy, self.zz = xp.meshgrid(self.x, self.x, self.x)
		self.r                    = xp.sqrt(self.xx * self.xx + self.yy * self.yy + self.zz * self.zz)
		r2_rolled                 = xp.roll (self.xx * self.xx + self.yy * self.yy + self.zz * self.zz, self.ngrid_h, axis=(0, 1, 2))
		self.Gamma                = (2 * xp.pi * self.a*2) ** (-1.5) * xp.exp(-0.5 * r2_rolled / self.a ** 2)
		self.nu_Gamma             = - r2_rolled / self.a**2 * self.Gamma
		self.Phi_p                = (self.Kp * self.beta / xp.pi) ** (1.5) * xp.exp(-self.beta * self.Kp * r2_rolled)
		self.Phi_s                = (self.Ks * self.beta / xp.pi) ** (1.5) * xp.exp(-self.beta * self.Ks * r2_rolled)
		self.f_L_p                = (1 - 2 * self.beta * self.Kp * r2_rolled / 3) * self.Phi_p
		self.f_L_s                = (1 - 2 * self.beta * self.Ks * r2_rolled / 3) * self.Phi_s

		# set up output file
		with open(self.operators_file, 'w') as f:
			f.write("# step | beta_mu_ex_S.real | beta_mu_ex_P.real | beta_P.real | rho_A.real | rho_B.real | rho_P.real | -kTlog(Qs) | -kTlog(Qp)\n")

		# set up the state file
		state_file = open(self.state_file, "wb")
		state_file.close()

		# set up the fields file
		field_file = open(self.field_file, "wb")
		field_file.close()
		
		return
	
	def initialize_fields(self):
		# this is the set of polymer chemical potential fields
		self.w_P       = xp.zeros(self.ngrid_dim, dtype=complex)
		self.w_P.real += xp.random.normal(size=self.ngrid_dim, scale=1)
		self.w_P.imag += xp.random.normal(size=self.ngrid_dim, scale=1)

		# this is the set of species A chemical potential fields
		self.w_A       = xp.zeros(self.ngrid_dim, dtype=complex)
		self.w_A.real += xp.random.normal(size=self.ngrid_dim, scale=1)
		self.w_A.imag += xp.random.normal(size=self.ngrid_dim, scale=1)

		# this is the set of species B chemical potential fields
		self.w_B       = xp.zeros(self.ngrid_dim, dtype=complex)
		self.w_B.real += xp.random.normal(size=self.ngrid_dim, scale=1)
		self.w_B.imag += xp.random.normal(size=self.ngrid_dim, scale=1)

		# this is the set of densities of each species
		self.rho_P     = xp.zeros(self.ngrid_dim, dtype=complex)
		self.rho_A     = xp.zeros(self.ngrid_dim, dtype=complex)
		self.rho_B     = xp.zeros(self.ngrid_dim, dtype=complex)

		# this is the set of the transformed densities that you need to compute the thermodynamic force
		self.eta_P     = xp.zeros(self.ngrid_dim, dtype=complex)
		self.eta_A     = xp.zeros(self.ngrid_dim, dtype=complex)
		self.eta_B     = xp.zeros(self.ngrid_dim, dtype=complex)

		# these are the propagator functions for the solvent and polymer
		self.qs_forwA = xp.zeros((2, )      + self.ngrid_dim, dtype=complex)
		self.qs_backB = xp.zeros((2, )      + self.ngrid_dim, dtype=complex)
		self.qp       = xp.zeros((self.dp,) + self.ngrid_dim, dtype=complex)

		return
	
	# this is the Langevin noise
	def noise(self):
		std = xp.sqrt(2 * self.dt / self.d3r * self.T)
		return std * xp.random.normal(size=self.ngrid_dim, loc=0, scale=1)
	
	# this is the convolution function
	def convolve(self, field_A, field_B):
		A_k = xp.fft.fftn(field_A)                 # fourier xfrm of A
		B_k = xp.fft.fftn(field_B)                 # fourier xfrm of B
		AB_k = A_k * B_k                           # product in fourier space
		field_AB = xp.fft.ifftn(AB_k) * self.d3r   # inversion of the fourier transform 
		return field_AB

	# this is the propagator for the diatomic solvent
	def solve_hetero_diatomic_propagator(self):
		self.qs_forwA[0] = xp.exp(-self.Omega_A)
		self.qs_forwA[1] = xp.exp(-self.Omega_B) * self.convolve(self.Phi_s, self.qs_forwA[0])
		self.qs_backB[0] = xp.exp(-self.Omega_B)
		self.qs_backB[1] = xp.exp(-self.Omega_A) * self.convolve(self.Phi_s, self.qs_backB[0])
		return
	
	# this is the propagator for the polymer
	def solve_homopolymer_propagator(self):
		self.qp[0] = xp.exp(-self.Omega_P)
		for j in range(1, self.dp):
			self.qp[j] = xp.exp(-self.Omega_P) * self.convolve(self.Phi_p, self.qp[j-1])
		return 
	
	# this is the homopolymer density operator
	def calc_homopolymer_density(self):
		self.rho_P *= 0.0
		for j in range(self.dp):
			self.rho_P += self.qp[j] * self.qp[self.dp - 1 - j]
		self.rho_P *= self.np * xp.exp(self.Omega_P) / (self.V * self.Qp)
		return
	
	# compute the density of the species in the diatomic solvent
	def calc_solvent_density(self):
		# in-place zero-ing out of the arrays 
		self.rho_A *= 0.0
		self.rho_B *= 0.0
		self.rho_A += self.qs_forwA[0] * self.qs_backB[1]
		self.rho_B += self.qs_backB[0] * self.qs_forwA[1]
		self.rho_A *= self.ns * xp.exp(self.Omega_A) / (self.V * self.Qs)
		self.rho_B *= self.ns * xp.exp(self.Omega_B) / (self.V * self.Qs)
		return 
	
	# compute the chemical potential of both, the solvent and the polymer
	def calc_chemical_potential(self):
		self.beta_mu_ex_S = - self.beta * self.ns * 2       * (self.us) / 2  - xp.log(self.Qs)
		self.beta_mu_ex_P = - self.beta * self.np * self.dp * (self.us) / 2  - xp.log(self.Qp)
		return

	def calc_pressure(self):
		# get the bulk free energy density: f_B
		self.f_B_P = -0.5 * self.convolve(self.Gamma + 2/3 * self.nu_Gamma, self.eta_P)
		self.f_B_A = -0.5 * self.convolve(self.Gamma + 2/3 * self.nu_Gamma, self.eta_A)
		self.f_B_B = -0.5 * self.convolve(self.Gamma + 2/3 * self.nu_Gamma, self.eta_B)

		# get the pressure contribution of coupling density with bulk free energy of each species 
		T1_P = 1 / self.V * xp.sum(self.rho_P * self.f_B_P) * self.d3r
		T1_A = 1 / self.V * xp.sum(self.rho_A * self.f_B_A) * self.d3r
		T1_B = 1 / self.V * xp.sum(self.rho_B * self.f_B_B) * self.d3r

		# set up the object to get the bonding free energy density: f_L
		q_fL_q = xp.zeros(self.ngrid_dim, dtype=complex)
		for j in range(self.dp-1):
			q_fL_q += self.qp[self.dp-2-j] * self.convolve(self.f_L_p, self.qp[j])

		# this is the pressure contribution from the bonding free energy term
		T2_P     = self.np / (self.V**2 * self.Qp) * self.d3r * xp.sum(q_fL_q)

		# set up the object to get the solvent bonding free energy
		q_fL_q  *= 0
		q_fL_q   = self.qs_forwA[0] * self.convolve(self.f_L_s, self.qs_forwA[0])

		# this is pressure contribution from the bonding free energy term
		T2_S     = self.ns / (self.V**2 * self.Qs) * self.d3r * xp.sum(q_fL_q)

		# ideal contribution is number of distinct molecules in box / volume
		betaP_id = (self.ns + self.np) / self.V

		# excess contribution adds up all the bulk and bonding pressure terms
		betaP_ex    = T1_P + T2_P + T1_A + T1_B + T2_S
		self.beta_P = betaP_id + betaP_ex
		return

	def calc_rg2(self, rho_P):
		net_mass = xp.sum(rho_P) * self.d3r
		xcm      = xp.sum(rho_P * self.xx) * self.d3r / net_mass
		ycm      = xp.sum(rho_P * self.yy) * self.d3r / net_mass
		zcm      = xp.sum(rho_P * self.zz) * self.d3r / net_mass
		Rcm      = xp.array([xcm, ycm, zcm]).real
		dx       = self.xx - Rcm[0]
		dy       = self.yy - Rcm[1]
		dz       = self.zz - Rcm[2]
		rg2      = xp.sum(rho_P * (dx * dx + dy * dy + dz * dz)) * self.d3r / net_mass
		rg       = xp.sqrt(rg2)
		return rg

	def run_complex_langevin(self):
		init_step = self.current_step + 1
		for step in range(self.current_step+1, self.current_step + 1 + self.nsteps):

			# update the current step
			self.current_step = step 

			# implement complex langevin dynamics here
			self.eta_A =	self.SL(self.Eval[0]) * self.Evec.T[0,0] * self.w_A + \
							self.SL(self.Eval[1]) * self.Evec.T[0,1] * self.w_B + \
							self.SL(self.Eval[2]) * self.Evec.T[0,2] * self.w_P
			
			self.eta_B =	self.SL(self.Eval[0]) * self.Evec.T[1,0] * self.w_A + \
							self.SL(self.Eval[1]) * self.Evec.T[1,1] * self.w_B + \
							self.SL(self.Eval[2]) * self.Evec.T[1,2] * self.w_P
			
			self.eta_P =	self.SL(self.Eval[0]) * self.Evec.T[2,0] * self.w_A + \
							self.SL(self.Eval[1]) * self.Evec.T[2,1] * self.w_B + \
							self.SL(self.Eval[2]) * self.Evec.T[2,2] * self.w_P
			
			# get the Omega fields
			self.Omega_A = self.convolve(self.Gamma, self.eta_A)
			self.Omega_B = self.convolve(self.Gamma, self.eta_B)
			self.Omega_P = self.convolve(self.Gamma, self.eta_P)

			# get the solvent partition function
			self.solve_hetero_diatomic_propagator()
			self.Qs = 1 / self.V * self.d3r * xp.sum(self.qs_forwA[2-1])

			# get the polymer partition function
			self.solve_homopolymer_propagator()
			self.Qp = 1 / self.V * self.d3r * xp.sum(self.qp[self.dp-1])

			# now start getting the density operators
			self.calc_solvent_density()
			self.calc_homopolymer_density()


			# get the energy differential
			f_A = xp.where(xp.abs(self.Eval[0])>1e-12, self.w_A/(self.beta * xp.abs(self.Eval[0])), 0)
			f_B = xp.where(xp.abs(self.Eval[1])>1e-12, self.w_B/(self.beta * xp.abs(self.Eval[1])), 0)
			f_P = xp.where(xp.abs(self.Eval[2])>1e-12, self.w_P/(self.beta * xp.abs(self.Eval[2])), 0)

			# get the partition function differential
			Q_A =   self.SL(self.Eval[0]) * self.Evec.T[0,0] * self.convolve(self.Gamma, self.rho_A) + \
					self.SL(self.Eval[0]) * self.Evec.T[0,1] * self.convolve(self.Gamma, self.rho_B) + \
					self.SL(self.Eval[0]) * self.Evec.T[0,2] * self.convolve(self.Gamma, self.rho_P)
			Q_B =   self.SL(self.Eval[1]) * self.Evec.T[1,0] * self.convolve(self.Gamma, self.rho_A) + \
					self.SL(self.Eval[1]) * self.Evec.T[1,1] * self.convolve(self.Gamma, self.rho_B) + \
					self.SL(self.Eval[1]) * self.Evec.T[1,2] * self.convolve(self.Gamma, self.rho_P)
			Q_P =   self.SL(self.Eval[2]) * self.Evec.T[2,0] * self.convolve(self.Gamma, self.rho_A) + \
					self.SL(self.Eval[2]) * self.Evec.T[2,1] * self.convolve(self.Gamma, self.rho_B) + \
					self.SL(self.Eval[2]) * self.Evec.T[2,2] * self.convolve(self.Gamma, self.rho_P)

			# calculate thermodynamic force
			dHdw_A  = f_A + Q_A
			dHdw_B  = f_B + Q_B
			dHdw_P  = f_P + Q_P

			# update fields
			self.w_A = self.w_A - self.dt * dHdw_A + self.noise()
			self.w_B = self.w_B - self.dt * dHdw_B + self.noise()
			self.w_P = self.w_P - self.dt * dHdw_P + self.noise()

			if xp.isnan(f_A).any():
				print(f"df_A is broken.", flush=True)
				exit()
			elif xp.isnan(f_B).any():
				print(f"df_B is broken.", flush=True)
				exit()
			elif xp.isnan(f_P).any():
				print(f"df_P is broken.", flush=True)
				exit()
			elif xp.isnan(Q_A).any():
				print(f"Q_A is broken.", flush=True)
				exit()
			elif xp.isnan(Q_B).any():
				print(f"Q_B is broken.", flush=True)
				exit()
			elif xp.isnan(Q_P).any():
				print(f"Q_P is broken.", flush=True)
				exit()


			# dump out operators 
			if (step % self.freq == 0):
				self.calc_pressure()
				self.calc_chemical_potential()
				print(f"step {step}: beta mu_ex_P = {self.beta_mu_ex_P:.4f}, beta mu_ex_S = {self.beta_mu_ex_S:.4f}, beta_P = {self.beta_P:.4f}", flush=True)
				with open(self.operators_file, 'a') as op_file:
					op_file.write(f"{step} | {self.beta_mu_ex_S.real:.4f} | {self.beta_mu_ex_P.real:.4f} | {self.beta_P.real:.4f} | {xp.mean(self.rho_A).real:.4f}" + \
					f"| {xp.mean(self.rho_B).real:.4f} | {xp.mean(self.rho_P).real:.4f} | {-self.T * xp.log(self.Qs):.4f} | {-self.T * xp.log(self.Qp):.4f}\n")

			# tack on more to the RHO FIELDS
			if step >= int(init_step + (self.nsteps/10)) and step % int(self.freq) == 0:
				self.RHO_A_FIELDS.append(self.rho_A)
				self.RHO_B_FIELDS.append(self.rho_B)
				self.RHO_P_FIELDS.append(self.rho_P)
				print(f"Total fields in A: {len(self.RHO_A_FIELDS)}, in B: {len(self.RHO_B_FIELDS)}, in P: {len(self.RHO_P_FIELDS)} at step #{step}.", flush=True)

			# dump out the state and RHO_FIELDS
			if len(self.RHO_A_FIELDS) > 100:
				print(f"Dumping out fields at #{step} and length is {len(self.RHO_A_FIELDS)}.", flush=True)
				self.dump_field_object()
				self.dump_state_object()
		
		# end of simulation
		return 

	# dump the state object
	def dump_state_object(self):
		with open(self.state_file, "wb") as f:
			pickle.dump(self, f)
		return

	# dump the field object
	def dump_field_object(self):
		fields = [self.RHO_A_FIELDS, self.RHO_B_FIELDS, self.RHO_P_FIELDS]
		with open(self.field_file, "ab") as f:
			pickle.dump(fields, f)
		self.RHO_A_FIELDS.clear()
		self.RHO_B_FIELDS.clear()
		self.RHO_P_FIELDS.clear()
		return

	def print_info(self):
		print(f"""
The interaction parameters are chi_AB = {self.chi_AB}, chi_AP = {self.chi_AP}, chi_BP = {self.chi_BP}.
The temperature is {self.T}, the volume of the system is {self.V} with edge length {self.L}.
The number of polymers in the system is {self.np} and the number of solvent particles is {self.ns}.
The degree of polymerization of the polymer is {self.dp} and solvents is 2.
The spring constant of polymer is {self.Kp} and the solvent is {self.Ks}.
The monomeric length scale is {self.a}.
The mesh is discretized into {self.mesh_fineness}^3 points. 
The size of the timestep is {self.dt} and number of steps to perform is {self.nsteps}.
The frequency of dumping output is {self.freq}.
The state will be dumped in {self.state_file}. Currently, the state is at step #{self.current_step}
The fields will be dumped in {self.field_file}.
The operators will be dumped in {self.operators_file}.
""", flush=True)
