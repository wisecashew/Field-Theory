#!/home/satyend/.conda/envs/FTS/bin/python

import cupy as xp
import pickle
import re

# this is the object per Josh Lequieu's formulation

class Blend:
	def __init__(self, input_file):
		self.read_input_file(input_file)
		self.current_step   = 0                         # this is the timestep counter
		self.RHO_A_FIELDS   = []                        # this is the object that will hold all the A density fields
		self.RHO_B_FIELDS   = []                        # this is the object that will hold all the B density fields
		return 

	def read_input_file(self, input_file):
		keys = ["u0", 'T', "Kp", "phiA", "phiB", "dA", "dB", 'L', \
				'a', "mesh_fineness", "dt", "nsteps", "rho0", \
				"output_freq", "field_file", "state_file", "operators_file"]
		pattern = r'=\s*(.+)'
		system = dict()
		with open(input_file, 'r') as file:
			for line in file:
				for key in keys:
					if re.match(r"^"+key+r"\s+", line):
						hit = re.search(pattern, line)
						system[key] = hit.group(1)
		
		# set up the parameters
		self.u0             = float(system["u0"])              # this is the cross interaction between species A and P
		self.T              = float(system["T"])               # this is the tempreature of the simulation
		self.Kp             = float(system["Kp"])              # this is the spring constant connecting the polymer beads
		self.np             = int(system["np"])                # this is the number of polymer molecules 
		self.dA             = int(system["dA"])                # this is the degree of polymerization
		self.dB             = int(system["dB"])                # this is the degree of polymerization
		self.phiA           = float(system["phiA"])            # this is the volume fraction of polymer A
		self.phiB           = float(system["phiB"])            # this is the volume fraction of polymer B
		self.rho0           = float(system["rho0"])            # this is the total particle density of the system
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

	def initialize_thermodynamic_geometric_parameters(self):
		self.beta          = 1 / self.T
		self.V             = self.L**3
		self.nA            = self.phiA * rho0 * self.V / self.dA
		self.nB            = self.phiB * rho0 * self.V / self.dB
		self.ngrid_dim     = (self.mesh_fineness, self.mesh_fineness, self.mesh_fineness)
		self.ngrid_h       = (self.ngrid_dim[0]//2, self.ngrid_dim[1]//2, self.ngrid_dim[2]//2)
		self.tot_grid      = self.ngrid_dim[0] * self.ngrid_dim[1] * self.ngrid_dim[2]
		self.d3r           = self.V / self.tot_grid
		self.rho           = xp.zeros(self.ngrid_dim, dtype=complex)
		self.q_fL_q        = xp.zeros(self.ngrid_dim, dtype=complex)
		self.q             = xp.zeros((self.dp,) + self.ngrid_dim, dtype=complex)	# this is the chain propagator
		return

	def initialize_grid(self):
		# this function initializes the grid over which we will evolve our fields
		self.x                    = xp.arange  (-self.L/2, self.L/2, self.L/self.ngrid_dim[0])
		self.dr                   = xp.asarray (self.x[1] - self.x[0])
		self.xx, self.yy, self.zz = xp.meshgrid(self.x, self.x, self.x)
		r2_rolled                 = xp.roll(self.xx * self.xx + self.yy * self.yy + self.zz * self.zz, self.ngrid_h, axis=(0, 1, 2))
		self.beta_us              = (self.beta * self.u0) / (8 * xp.pi**1.5 * self.a**3)
		self.Gamma                = (2 * xp.pi * self.a**2)**-1.5 * xp.exp(-0.5 * r2_rolled / self.a**2)
		self.nu_Gamma             = - r2_rolled / self.a**2 * self.Gamma
		self.Phi_p                = (self.Kp * self.beta / xp.pi)**(1.5) * xp.exp(-self.beta * self.Kp * r2_rolled)
		self.f_L                  = (1 - 2 * self.beta * self.Kp * r2_rolled / 3) * self.Phi_p

		# setup output file
		with open(self.operators_file, 'w') as f:
			f.write("# step beta_mu_ex.real beta_mu_ex.imag beta_P.real beta_P.imag rho.real -kTlog(Q)\n")
		return

	def SL(self, eigenval):
		sl = 1 if eigenval < 0 else 1j
		return sl

	def initialize_fields(self):
		self.w_plus        = xp.zeros(self.ngrid_dim, dtype=complex)
		self.w_plus.real  += xp.random.normal(size=self.ngrid_dim, scale=1)
		self.w_plus.imag  += xp.random.normal(size=self.ngrid_dim, scale=1)

		self.w_minus       = xp.zeros(self.ngrid_dim, dtype=complex)
		self.w_minus.real += xp.random.normal(size=self.ngrid_dim, scale=1)
		self.w_minus.imag += xp.random.normal(size=self.ngrid_dim, scale=1)
		return

	def noise(self):
		std = xp.sqrt(2 * self.dt / self.d3r * self.T) # this is the Langevin sampling
		return std * xp.random.normal(size=self.ngrid_dim, loc=0, scale=1)

	def convolve(self, field_A, field_B):
		A_k = xp.fft.fftn(field_A)                 # fourier xfrm of A
		B_k = xp.fft.fftn(field_B)                 # fourier xfrm of B
		AB_k = A_k * B_k                           # product in fourier space
		field_AB   = xp.fft.ifftn(AB_k) * self.d3r # inversion of the fourier transform
		return field_AB

	def solve_propagator(self):
		self.qA[0] = xp.exp(-self.OmegaA)
		self.qB[0] = xp.exp(-self.OmegaB)
		for j in range(1, self.dp):
			self.qA[j] = xp.exp(-self.OmegaA) * self.convolve(self.Phi_p, self.qA[j-1])
			self.qB[j] = xp.exp(-self.OmegaB) * self.convolve(self.Phi_p, self.qB[j-1])
		return

	def calc_density(self):
		self.rhoA *= 0.0
		self.rhoB *= 0.0
		for j in range(0, self.dp):
			self.rhoA += self.qA[j] * self.q[self.dA - 1 -j]
			self.rhoB += self.qB[j] * self.q[self.dB - 1 -j]
		self.rhoA *= self.nA * xp.exp(self.OmegaA) / (self.V * self.QA)
		self.rhoB *= self.nB * xp.exp(self.OmegaB) / (self.V * self.QB)
		return

	def calc_pressure(self):
		fB = -0.5 * self.convolve(self.Gamma + 2/3 * self.nu_Gamma, self.eta_A)
		T1 = self.dp / self.rho0 * self.d3r * xp.sum(self.rho * fB)
		self.q_fL_q *= 0.0
		for j in range(0, self.dp - 1):
			self.q_fL_q += self.q[self.dp - 2 -j] * self.convolve(self.f_L, self.q[j])
		T2 = 1 / self.Q * self.d3r * xp.sum(self.q_fL_q)
		self.beta_P = self.np / self.V + self.np / self.V ** 2 * (T1 + T2)
		return

	def calc_chemical_potential(self):
		self.beta_muA_ex = -0.5 * self.beta_us * self.dA - xp.log(self.QA)
		self.beta_muB_ex = -0.5 * self.beta_us * self.dB - xp.log(self.QB)
		return

	def run_complex_langevin(self):
		init_step = self.current_step + 1
		for step in range(self.current_step+1, self.current_step + self.nsteps):
			print(f"@step {step}.", flush=True)
			# update current step
			self.current_step = step
			
			# implement complex langevin dynamics here
			self.OmegaA = self.convolve(self.Gamma, 1j*self.w_plus - self.w_minus)
			self.OmegaB = self.convolve(self.Gamma, 1j*self.w_plus + self.w_minus)
			self.solve_propagator()
			self.QA     = 1/self.V * self.d3r * xp.sum(xp.exp(-self.OmegaA))
			self.QB     = 1/self.V * self.d3r * xp.sum(xp.exp(-self.OmegaB))
			self.calc_density()

			# calculate the thermodynamic force
			dHdw_plus  = 2 * rho0 / (self.chi + 2 * beta_u0 * rho0) * self.w_plus + 1j * self.convolve(self.Gamma, self.rhoA + self.rhoB)
			dHdw_minus = 2 * rho0 / self.chi * self.w_minus                       + 1  * self.convolve(self.Gamma, -self.rhoA + self.rhoB)

			# update fields
			self.w_plus  = self.w_plus  - self.dt * dHdw_plus  + self.noise()
			self.w_minus = self.w_minus - self.dt * dHdw_minus + self.noise()

			# dump out operators
			if (step % self.freq == 0):
				self.calc_pressure()
				self.calc_chemical_potential()
				print(f"{step = }\t{self.beta_muex = :.4f}\t{self.beta_P = :.4f}", flush=True)
				with open(self.operators_file, 'a') as f:
					f.write(f"{step} {self.beta_muA_ex.real:.2f} {self.beta_muB_ex.imag:.2f} {self.beta_P.real:.2f} {xp.mean(self.rhoA):.2f} {-self.T * xp.log(self.QA):.2f}\n")
			if step >= int(init_step + self.nsteps/10) and step % int(self.freq) == 0:
				self.RHO_A_FIELDS.append(self.rhoA)
				self.RHO_B_FIELDS.append(self.rhoB)

		return
	
	def dump_field_object(self):
		with open(self.field_file, 'wb') as f:
			pickle.dump(self, f)
		return

