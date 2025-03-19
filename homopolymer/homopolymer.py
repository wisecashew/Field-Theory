#!/home/satyend/.conda/envs/FTS/bin/python

import cupy as xp
import pickle

class Homopolymer:
	def __init__(self, system):
		self.u0             = system["u0"]              # this is the contact energy
		self.Kp             = system["Kp"]              # this is the spring constant for the polymer bonds
		self.Dp             = system["Dp"]              # this is the degree of polymerization
		self.T              = system["T"]               # this is the temperature of the system
		self.rho0           = system["rho0"]            # this is the number of particles in the system
		self.L              = system["L"]               # this is the length of the simulation box
		self.a              = system["a"]               # this is the atomic diameter
		self.b0             = system["b0"]              # this is the bond length b0
		self.mesh_fineness  = system["mesh_fineness"]   # this is the fineness of the mesh
		self.dt             = system["dt"]              # this is the size of the timestep
		self.nsteps         = system["nsteps"]          # this is the number of timesteps to perform
		self.freq           = system["freq"]            # this is the output dump frequency
		self.field_file     = system["field_file"]      # this is the file in which the Homopolymer object will be pickled
		self.operators_file = system["operators_file"]  # this is the file where thermodynamic information will be output
		self.current_step   = 0                         # this is the timestep counter
		self.RHO_FIELDS     = []                        # this is the object that will hold all the density fields
		return 

	def set_radial_properties(self):
		# calculate the radial properties of the system
		self.rdf_count    = 0
		self.Rmax         = self.L / 2
		self.dr           = self.L / 100
		self.r_range      = xp.linspace(0, self.Rmax+self.dr, num=int(self.Rmax/self.dr) + 1)
		self.bin_centers  = 0.5 * (self.r_range[1:] + self.r_range[:-1])
		xx, yy, zz        = xp.meshgrid(self.x, self.x, self.x)
		self.r2           = xp.sqrt(xx**2 + yy**2 + zz**2)
		self.r2_flat      = self.r2.flatten()
		self.mask         = self.r2_flat <= self.Rmax

		# take the r of each voxel and put it in the appropriate bin
		# if  r_range[j] <= r2_flat[i] < r_range[j+1], 
		# self.bin_indices[i] tells me in which bin of self.r_range the element self.r2_flat goes.
		self.bin_indices  = xp.digitize(self.r2_flat[self.mask], self.r_range) - 1

		# setting up the spherical correlation array
		self.sphere_corr  = xp.zeros(len(self.r_range) - 1)
		return

	def initialize_thermodynamic_geometric_parameters(self):
		self.beta          = 1 / self.T
		self.V             = self.L**3
		self.n             = self.rho0 * self.V / self.Dp
		self.ngrid_dim     = (self.mesh_fineness, self.mesh_fineness, self.mesh_fineness)
		self.ngrid_h       = (self.ngrid_dim[0]//2, self.ngrid_dim[1]//2, self.ngrid_dim[2]//2)
		self.tot_grid      = self.ngrid_dim[0] * self.ngrid_dim[1] * self.ngrid_dim[2]
		self.d3r           = self.V / self.tot_grid
		self.rho           = xp.zeros(self.ngrid_dim, dtype=complex)
		self.q_fL_q        = xp.zeros(self.ngrid_dim, dtype=complex)
		self.q             = xp.zeros((self.Dp,) + self.ngrid_dim, dtype=complex)				# this is the chain propagator
		return

	def initialize_grid(self):
		# this function initializes the grid over which we will evolve our fields
		self.x        = xp.arange  (-self.L/2, self.L/2, self.L/self.ngrid_dim[0])
		self.dr       = xp.asarray (self.x[1] - self.x[0])
		xx, yy, zz    = xp.meshgrid(self.x, self.x, self.x)
		r2_rolled     = xp.roll(xx * xx + yy * yy + zz * zz, self.ngrid_h, axis=(0, 1, 2))
		self.beta_us  = (self.beta * self.u0) / (8 * xp.pi**1.5 * self.a**3)
		self.Gamma    = (2 * xp.pi * self.a**2)**-1.5 * xp.exp(-0.5 * r2_rolled / self.a**2)
		self.nu_Gamma = - r2_rolled / self.a**2 * self.Gamma
		self.Phi_p    = (self.Kp * self.beta / xp.pi)**(1.5) * xp.exp(-self.beta * self.Kp * r2_rolled)
		self.fL       = (1 - r2_rolled / self.b0 ** 2) * self.Phi_p
		self.set_radial_properties()

		# setup output file
		with open(self.operators_file, 'w') as f:
			f.write("# step beta_mu_ex.real beta_mu_ex.imag beta_P.real beta_P.imag rho.real -kTlog(Q)\n")
		return

	def initialize_fields(self):
		self.w       = xp.zeros(self.ngrid_dim, dtype=complex)
		self.w.real += xp.random.normal(size=self.ngrid_dim, scale=1)
		self.w.imag += xp.random.normal(size=self.ngrid_dim, scale=1)
		return
	
	def noise(self):
		std = xp.sqrt(2 * self.dt / self.d3r)
		return std * xp.random.normal(size=self.ngrid_dim, loc=0, scale=1)

	def convolve(self, field_A, field_B):
		A_k = xp.fft.fftn(field_A)                 # fourier xfrm of A
		B_k = xp.fft.fftn(field_B)                 # fourier xfrm of B
		AB_k = A_k * B_k                           # product in fourier space
		field_AB   = xp.fft.ifftn(AB_k) * self.d3r # inversion of the fourier transform
		return field_AB

	def solve_propagator(self):
		self.q[0] = xp.exp(-self.Omega)
		for j in range(1, self.Dp):
			self.q[j] = xp.exp(-self.Omega) * self.convolve(self.Phi_p, self.q[j-1])
		return

	def calc_density(self):
		self.rho *= 0.0
		for j in range(0, self.Dp):
			self.rho += self.q[j] * self.q[self.Dp - 1 -j]
		self.rho *= self.n * xp.exp(self.Omega) / (self.V * self.Q)
		return

	def calc_pressure(self):
		fB = -0.5 * self.convolve(self.Gamma + 2/3 * self.nu_Gamma, self.w)
		T1 = self.Dp / self.rho0 * self.d3r * xp.sum(self.rho * fB)
		self.q_fL_q *= 0.0
		for j in range(0, self.Dp - 1):
			self.q_fL_q += self.q[self.Dp - 2 -j] * self.convolve(self.fL, self.q[j])
		T2 = 1 / self.Q * self.d3r * xp.sum(self.q_fL_q)
		self.beta_P = self.n / self.V + self.n / self.V ** 2 * (T1 + T2)
		return
	
	def calc_chemical_potential(self):
		self.beta_muex = -0.5 * self.beta_us * self.Dp - xp.log(self.Q)
		return

	def run_complex_langevin(self):
		for step in range(self.current_step+1, self.current_step + self.nsteps):
			print(f"@step {step}.", flush=True)
			# update current step
			self.current_step = step
			
			# implement complex langevin dynamics here
			self.Omega = 1j * self.convolve(self.Gamma, self.w)
			self.solve_propagator()
			self.Q     = 1/self.V * self.d3r * xp.sum(xp.exp(-self.Omega))
			self.calc_density()

			# calculate the thermodynamic force
			dHdw = self.w / (self.beta * self.u0) + 1j * self.convolve(self.Gamma, self.rho)

			# update fields
			self.w = self.w - self.dt * dHdw + self.noise()

			# dump out operators
			if (step % self.freq == 0):
				self.calc_pressure()
				self.calc_chemical_potential()
				print(f"{step = }\t{self.beta_muex = :.4f}\t{self.beta_P = :.4f}", flush=True)
				with open(self.operators_file, 'a') as f:
					f.write(f"{step} {self.beta_muex.real:.2f} {self.beta_muex.imag:.2f} {self.beta_P.real:.2f} {self.beta_P.imag:.2f} {xp.mean(self.rho.real):.2f} {-self.T * xp.log(self.Q):.2f}\n")
			if step >= self.nsteps/10 and step % int(self.freq) == 0:
				self.RHO_FIELDS.append(self.rho)

		return
	
	def dump_field_object(self):
		with open(self.field_file, 'wb') as f:
			pickle.dump(self, f)
		return
	
