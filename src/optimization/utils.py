import numpy as np
import matplotlib.pyplot as plt

def plot_ellipse(ax, radius, pos=[0,0,0], Ndisc=10, zorder=-1, label=None):
	"""
	Create a 3D visualization of the moon with its surface texture from a local image file.
	"""
	
	if radius.ndim == 0: 
		radius= np.array([radius, radius, radius])

	# Convert the image to a numpy array and normalize
	phi   = np.linspace(0, np.pi, Ndisc)  # tune this to get a better resolution
	theta = np.linspace(0, 2*np.pi, 2*Ndisc)
	phi, theta = np.meshgrid(phi, theta)

	x = radius[0] * np.sin(phi) * np.cos(theta)
	y = radius[1] * np.sin(phi) * np.sin(theta)
	z = radius[2] * np.cos(phi)

	# Get RGB colors # Normalize the image colors to [0,1] range for matplotlib
	surf = ax.plot_surface(
		x, y, z,
		rstride=1, cstride=1,
		color='red',            # single face color
		edgecolor=None,        # outline color
		linewidth=0.1,
		antialiased=True,
		shade=False,               # keep faces purely white (disable shading), 
		alpha =0.2,
		zorder=zorder,
		label=label,
	)
	return ax
