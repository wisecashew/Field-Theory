#!/home/satyend/.conda/envs/FTS/bin/python

# Joshua Lequieu 5/20/2024

#import numpy as xp # uncomment to run on cpu
import cupy as xp # uncomment to run on gpu

def main():

	# model parameters
	beta_u0 = 0.007           # excluded volume strength
	rho0 = 2                  # overall density
	L = 20.0                  # box length
	a = 0.5                   # units of b
	NA = 50                   # polymer A length
	NB = 50                   # polymer B length
	chi = 0.08                # Flory-Huggins interaction parameter
	phiA = 0.5                # volume fraction of polymer A
	phiB = 1 - phiA           # volume fraction of polymer B
	b = 1                     # statistical segment length (equal for A and B)
	V = L**3                  # box volume
	nA = phiA * rho0 * V / NA # number of A polymers
	nB = phiB * rho0 * V / NB # number of B polymers

	npw = (int(L//a), int(L//a), int(L//a))       # number of grid points in each dimension
	npw_half = (npw[0]//2, npw[1]//2, npw[2]//2)  # half the grid points in each dimension
	M = npw[0] * npw[1] * npw[2]                  # total number of grid points
	d3r = V / M                                   # volume of one voxel
  
	# other parameters
	dt = 0.001                                    # timestep
	nsteps = int(5e4)                             # number of steps to run FTS simulation for
	freq = int(1e2)                               # frequency to write output

	# setup output file
	output_filename = 'operators.dat'
	with open(output_filename, 'w') as f:
		f.write("# step beta_muA_ex.real beta_muA_ex.imag beta_muB_ex.real beta_muB_ex.imag beta_P.real beta_P.imag\n")

	# setup r2 grid
	x = xp.arange(-L/2, L/2, L/npw[0])
	xx,yy,zz = xp.meshgrid(x,x,x)
	r2 = xp.roll(xx*xx + yy*yy + zz*zz, npw_half, axis=(0,1,2))
 
	# precompute variables
	beta_us = beta_u0 / (8 * xp.pi**1.5 * a**3)
	Gamma = (2 * xp.pi * a**2)**-1.5 * xp.exp(-0.5 * r2 / a**2)
	nu_Gamma = - r2 / a**2 * Gamma
	Phi = (3 / (2 * xp.pi * b**2))**1.5 * xp.exp(-1.5 * r2 / b**2)
	fL = (1 - r2 / b**2) * Phi
  
	# initialize w fields
	w_plus = xp.zeros(npw, dtype=complex)
	w_minus = xp.zeros(npw, dtype=complex)
	for field in [w_plus, w_minus]:
		field.real += xp.random.normal(size=npw, scale=1)
		field.imag += xp.random.normal(size=npw, scale=1)
	rhoA_avg = xp.zeros(npw, dtype=complex)
	rhoB_avg = xp.zeros(npw, dtype=complex)

	# main FTS loop 
	print(f"Running FTS using {xp.__name__}")
	for step in range(0,nsteps):
		OmegaA = convolve(Gamma, 1j*w_plus + w_minus, d3r)
		OmegaB = convolve(Gamma, 1j*w_plus - w_minus, d3r)

		# solve propagator
		qA = solve_propagator( NA, V, OmegaA, Phi)
		qB = solve_propagator( NB, V, OmegaB, Phi)

		QA = 1 / V * d3r * xp.sum(qA[NA-1])   # single molecule partition function
		QB = 1 / V * d3r * xp.sum(qB[NB-1])   # single molecule partition function

		# compute density field
		rhoA = calc_density( nA, NA, V, OmegaA, qA, QA)
		rhoB = calc_density( nB, NB, V, OmegaB, qB, QB)
		rhoA_avg += rhoA
		rhoB_avg += rhoB

		# calculate operators
		beta_muAex = - 0.5*beta_us*NA - xp.log(QA)   # excess chemical potential
		beta_muBex = - 0.5*beta_us*NB - xp.log(QB)   # excess chemical potential
		beta_P = calc_pressure( nA, NA, V, Gamma, nu_Gamma, fL, 1j*w_plus - w_minus, qA, QA, rhoA) \
			   + calc_pressure( nB, NB, V, Gamma, nu_Gamma, fL, 1j*w_plus + w_minus, qB, QB, rhoB) 

		# calculate thermodynamic forces
		dHdw_plus = 2*rho0 / (chi + 2 * beta_u0 * rho0) * w_plus + 1j*convolve(Gamma, rhoA + rhoB, d3r)  
		dHdw_minus = 2*rho0 / chi * w_minus + convolve(Gamma, - rhoA + rhoB, d3r)  

		# update fields
		w_plus = w_plus - dt*dHdw_plus + noise(dt, npw, d3r)
		w_minus = w_minus - dt*dHdw_minus + noise(dt, npw, d3r) 

		# output to stdout and to file
		if ((step % freq) == 0):
			print(f"{step = }\t{beta_muAex = :.4f}\t{beta_muBex = :.4f}\t{beta_P = :.4f}") 
			with open(output_filename,'a') as f:
				f.write(f"{step} {beta_muAex.real} {beta_muAex.imag} {beta_muBex.real} {beta_muBex.imag} {beta_P.real} {beta_P.imag}\n")
		
		# plot and then zero block averages of rhoA and rhoB
		#plot_fields(xx, yy, zz, fields=[rhoA_avg / freq, rhoB_avg / freq], labels=['<rhoA>','<rhoB>'], filename=f'rho_{step}.png')
		rhoA_avg = xp.zeros(npw, dtype=complex)
		rhoB_avg = xp.zeros(npw, dtype=complex)

	# finally plot fields at final timestep
	# plot_fields(xx, yy, zz, fields=[rhoA_avg / freq, rhoB_avg / freq], labels=['<rhoA>','<rhoB>'], filename=f'rho.png')

def convolve(x,y,d3r):
	''' perfom a spatial convolution between two fields x and y'''
	return xp.fft.ifftn( xp.fft.fftn(x) * xp.fft.fftn(y)) * d3r

def noise(dt, npw, d3r):
	''' compute fts random noise '''
	std = xp.sqrt(2 * dt / d3r)
	return std * xp.random.randn(*npw) 

def solve_propagator(N, V, Omega, Phi):
	''' solve propagator q '''
	npw = Omega.shape                      # number of grid points in each dimension
	d3r = V / (npw[0] * npw[1] * npw[2])   # volume of each voxel
	q = xp.zeros((N,) + npw, dtype=complex)
	q[0] = xp.exp(-Omega)
	for j in range(1,N):
		q[j] = xp.exp(-Omega) * convolve(Phi, q[j-1], d3r)
	return q

def calc_density(n, N, V, Omega, q, Q):
	''' calc density field rho '''
	npw = Omega.shape                     # number of grid points in each dimension
	rho = xp.zeros(npw, dtype=complex)
	for j in range(N):
		rho += q[j] * q[N-1-j]
	rho *= n * xp.exp(Omega) / V / Q
	return rho

def calc_pressure(n, N, V, Gamma, nu_Gamma, fL, w, q, Q, rho):
	''' calculate pressure operator '''
	npw = Gamma.shape                                   # number of grid points in each dimension
	rho0 = n * N / V                                    # overall density
	d3r = V / (npw[0] * npw[1] * npw[2])                # volume of each voxel
	fB = -0.5 * convolve(Gamma + 2/3*nu_Gamma, w, d3r)  # bead function
	T1 = N / rho0 * d3r * xp.sum(rho * fB)
	q_fL_q = xp.zeros(npw, dtype=complex)
	for j in range(N-1):
		q_fL_q += q[N-2-j] * convolve(fL, q[j], d3r)
	T2 = 1 / Q * d3r * xp.sum(q_fL_q) 
	beta_P = n / V + n / V**2 * (T1 + T2)               # total pressure
	return beta_P

def write_particle_params(beta_u0, chi, rho0,  a, b):
	''' write model parameters to a json file '''
	params = {}
	params['pair'] = {}
	params['pair']['use_coulomb'] = False
	params['pair']['type'] = 'gaussian'
	eps_AA = eps_BB = beta_u0 / (8*xp.pi**1.5 * a**3)
	eps_AB = (beta_u0 + chi/rho0) / (8*xp.pi**1.5 * a**3)
	params['pair']['gauss_eps'] = [ eps_AA, eps_AB, eps_BB ]
	sig = float(xp.sqrt(2 * a**2)) 
	params['pair']['gauss_sig'] = [ sig, sig, sig ]
	params['pair']['rcut'] = [ 6.4*a, 6.4*a, 6.4*a ]
	params['bond'] = {}
	params['bond']['kbond'] = [ 3/b**2, 3/b**2, 3/b**2 ]
	params['bond']['r0'] = [0.0, 0.0, 0.0]
	params['bond']['type'] = "harmonic"
	params['species'] = {}
	params['species']['labels'] = ['A', 'B']

	with open('particle.json','w') as f:
		json.dump(params, f, indent=4)

if __name__ == '__main__': 
	main()