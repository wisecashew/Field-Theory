#!/home/satyend/.conda/envs/FTS/bin/python

import cupy as xp
import pickle

class Monatomic_Fluid:
	def __init__(self, system):
		self.u0             = system["u0"]
		self.T              = system["T"]
		self.rho0           = system["rho0"]
		self.L              = system["L"]
		self.a              = system["a"]
		self.mesh_fineness  = system["mesh_fineness"]
		self.dt             = system["dt"]
		self.nsteps         = system["nsteps"]
		self.freq           = system["freq"]
		self.field_file     = system["field_file"]
		self.operators_file = system["operators_file"]
		self.current_step   = 0
		self.RHO_FIELDS     = []
		return 

	def set_radial_properties(self):
		# calculate the radial properties of the system
		self.rdf_count    = 0
		self.Rmax         = self.L / 2
		self.dr           = self.L/100
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
		self.n             = int(self.rho0 * self.V)
		self.ngrid_dim     = (self.mesh_fineness, self.mesh_fineness, self.mesh_fineness)
		self.ngrid_h       = (self.ngrid_dim[0]//2, self.ngrid_dim[1]//2, self.ngrid_dim[2]//2)
		self.tot_grid      = self.ngrid_dim[0] * self.ngrid_dim[1] * self.ngrid_dim[2]
		self.d3r           = self.V / self.tot_grid
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

	def calc_pressure(self):
		fB          = -0.5 * (1j * self.convolve(self.Gamma + 2/3 * self.nu_Gamma, self.w))
		self.beta_P = self.n / self.V + self.n / self.V**2 / self.rho0 * self.d3r * xp.sum(self.rho * fB)
		return
	
	def calc_chemical_potential(self):
		self.beta_muex = -0.5 * self.beta_us - xp.log(self.Q)
		return

	def run_complex_langevin(self):
		init_step = self.current_step + 1
		for step in range(self.current_step+1, self.current_step + 1 + self.nsteps):
			
			# update current step
			self.current_step = step
			
			# implement complex langevin dynamics here
			self.Omega = 1j * self.convolve(self.Gamma, self.w)
			self.Q     = 1/self.V * self.d3r * xp.sum(xp.exp(-self.Omega))
			self.rho   = self.rho0 / self.Q  * xp.exp(-self.Omega)

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
			if step >= int(init_step + (self.nsteps)/10) and step % int(self.freq) == 0:
				self.RHO_FIELDS.append(self.rho)

		return
	
	def dump_field_object(self):
		with open(self.field_file, 'wb') as f:
			pickle.dump(self, f)
		return
	
	def print_information(self):
		# print out the information
		print(f"""
		The contact potential is u0 is {self.u0}.
		The temperature of the system is {self.T}, density is {self.rho0}, volume is {self.V}.
		Therefore, the number of particles in the system is {self.n}.
		The complex Langevin sampling will be run for {self.nsteps} steps with a step size of {self.dt}.
		The output will be dumped out every {self.freq} steps. 
		The on-the-fly computations will be in {self.operators_file}.
		The final state object will be in {self.field_file}.
		""", flush=True)
		return
