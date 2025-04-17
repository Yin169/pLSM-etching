import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io, transform
from multiprocessing import Pool


def grad(x):
    return np.array(np.gradient(x))

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

def stopping_fun(x):
	return 1. / (1. + norm(grad(x))**2)



class TimeScheme(object):
	def __init__(self, dt):
		self.dt = dt

	def EulerForward(self, u, rhs, i, j):
		return u[i, j] - self.dt * rhs(u, i, j)

class SpatialScheme:
	@staticmethod
	def Upwind(u, i, j, ii, jj):
		return u[i, j] - u[ii, jj]


class DualTimeStepping:
	def __init__(self, phi, dt, nt, spatialsch, F_v):
		self.phi = phi
		self.nx = phi.shape[0]
		self.ny = phi.shape[1]
		self.dx = 1.0/(self.nx-1)
		self.dy = 1.0/(self.ny-1)
		self.dt = dt  # physical time step
		self.nt = nt  # number of physical time steps
		self.sudot = 1e-3  # time step for pseudo-time iterations
		self.gamma = 0.5
		self.max_pseudo_iter = 2000  # maximum pseudo-time iterations
		self.pseudo_tol = 1e-2  # convergence tolerance for pseudo-time iterations
		self.spatialsch = spatialsch
		self.initialize()
		self.F_v = F_v

	def initialize(self):
		for i in range(self.nx):
			for j in range(self.ny):
				# self.phi[i,j]=np.sqrt((i-self.nx/2)*(i-self.nx/2)+(j-self.ny/2)*(j-self.ny/2))-(self.nx/4)
				self.phi[i, j] = max(self.nx/4 - i,  i - self.nx/2, self.ny/4 - j, j - self.ny/2)

	def F(self, phi, i, j):
		return self.F_v[i,j]

	def rhs(self, phi, i, j):
		delt_xn = self.spatialsch(phi, i, j, i-1, j)
		delt_yn = self.spatialsch(phi, i, j, i, j-1)
		delt_xp = self.spatialsch(phi, i+1, j, i, j)
		delt_yp = self.spatialsch(phi, i, j+1, i, j)
		deltp = max(delt_xn/self.dx, 0)**2 + max(delt_yn/self.dy, 0)**2 + min(delt_xp/self.dx, 0)**2 + min(delt_yp/self.dy, 0)**2
		deltn = max(delt_xp/self.dx, 0)**2 + max(delt_yp/self.dy, 0)**2 + min(delt_xn/self.dx, 0)**2 + min(delt_yn/self.dy, 0)**2
		F = self.F(phi, i,j)
		return max(F, 0) * np.sqrt(deltp) + min(F, 0) * np.sqrt(deltn)

	def residual(self, phi_n, phi_m, phi_nm1, i, j):
		return ((1 + self.gamma) * (phi_m[i, j] - phi_n[i, j]) - self.gamma * (phi_n[i,j] - phi_nm1[i,j]))/self.dt + self.rhs(phi_m, i, j)
	

	def dRdphi(self, phi, i, j):
		# Simplified and corrected Jacobian approximation
		# This is a numerical approximation of the derivative of the residual with respect to phi
		epsilon = 1e-6
		phi_perturbed = phi.copy()
		phi_perturbed[i, j] += epsilon

		res1 = self.rhs(phi, i, j)
		res2 = self.rhs(phi_perturbed, i, j)
		
		return (res2 - res1) / epsilon	

	def Coe(self, phi_m, i, j):
		return 1/self.sudot + (1+self.gamma) / self.dt + self.dRdphi(phi_m, i, j) 

	def wrap_func(self, args):
		phi_n, phi_m, phi_nm1, i, ny = args
		return [-self.residual(phi_n, phi_m, phi_nm1, i, j) / self.Coe(phi_m, i, j) for j in range(1, ny-1)]

	def solve(self):
		phi_m = self.phi.copy()
		phi_n = self.phi.copy()
		phi_nm1 = self.phi.copy()

		for t in range(self.nt):
			# plt.figure(figsize=(4,4))
			plt.contour(self.phi, levels=[0], colors='r')
			io.imshow(img_smooth)
			plt.title(f"time={t*self.dt}")
			# plt.xlabel("x")
			# plt.ylabel("y")
			plt.savefig(f"./out/phi_{t}.png")
			plt.clf()
			
			# Store previous solutions properly
			phi_nm1 = phi_n.copy()
			phi_n = self.phi.copy()
			
			self.phi = self.apply_boundary_conditions(self.phi)
			phi_m = self.phi.copy()
			
			# Pseudo-time iterations
			for pst in range(self.max_pseudo_iter):
				delta_m = np.zeros((self.nx, self.ny))
				with Pool(6) as pool:
					args = [(phi_n, phi_m, phi_nm1, i, self.ny) for i in range(1, self.nx-1)]
					ans = pool.map(self.wrap_func, args)
					delta_m[1:-1, 1:-1] = np.array(ans).reshape(self.nx-2, self.ny-2)
				
				# Apply update with relaxation for stability 
				phi_m += delta_m
				phi_m = self.apply_boundary_conditions(phi_m)
				
				# Check for convergence
				residual_norm = np.linalg.norm(delta_m)
				if pst % 1 == 0:
					print(f"Pseudo-time step {pst+1}: Residual norm = {residual_norm}")
				if residual_norm < self.pseudo_tol:
					break
			
			self.phi = phi_m.copy()


			
	
	def apply_boundary_conditions(self, phi):
		# Example boundary conditions (adjust as needed)
		# Zero gradient at boundaries
		phi[0, :] = phi[1, :]
		phi[-1, :] = phi[-2, :]
		phi[:, 0] = phi[:, 1]
		phi[:, -1] = phi[:, -2]
		return phi
	
		
if __name__ == '__main__':
	img = io.imread('data/DRIVE/training/images/21_training.tif')

	img = img - np.mean(img)
	img = color.rgb2gray(img)
	img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma=1)
	img_smooth = transform.resize(img_smooth, (56, 56))

	F_v = stopping_fun(img_smooth)
	print(img.shape)

	phi = np.zeros(img_smooth.shape)
	spatialsch = SpatialScheme
	solver = DualTimeStepping(phi, 0.01, 20000, spatialsch.Upwind, F_v)
	solver.solve()