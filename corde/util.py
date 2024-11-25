import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import csr_matrix
from scipy.special import erfinv
from .constants import C

def inner_product(x, y):
    return np.einsum('...i,...i', x, y)

def norm(x):
    return np.sqrt(inner_product(x,x))

def normalize(x):
    
    if x.ndim == 1:
        x = x.reshape(1,x.shape[0])
        
    return np.squeeze(x/norm(x).reshape(x.shape[0],1))

def cosine(x,y):
    return inner_product(x,y)/(norm(x)*norm(y))

def rotate(x, y, theta):
    """Rotate vector x about vector y by theta radians."""
    # Refer: http://math.stackexchange.com/a/1432182/81266
    if x.ndim == 1:
        x = x.reshape(1,x.shape[0])
    if y.ndim == 1:
        y = y.reshape(1,y.shape[0])
    if np.isscalar(theta):
        theta = np.array([theta])
        
    par = (inner_product(x, y) / inner_product(y, y)).reshape(x.shape[0],1) * y
    perp = x - par
    w = np.cross(y, perp)
    rot_x = par + perp * np.cos(theta).reshape(theta.shape[0],1) + norm(perp).reshape(x.shape[0],1) * normalize(w) * np.sin(theta).reshape(theta.shape[0],1)
    
    return np.squeeze(rot_x)
    
def dmax(del_t, xi, pi, z, theta, phi):
    p_norm = normalize(pi)
    x0 = POFI(z, theta, phi)
    return ((C**2) * (del_t)**2 - inner_product(xi-x0, xi-x0))/(2*(C*(del_t)-inner_product(xi-x0, p_norm)))
    
def POFI(POFI_z, theta, phi):
    POFI_x = - POFI_z * np.tan(theta) * np.cos(phi)
    POFI_y = - POFI_z * np.tan(theta) * np.sin(phi)
    
    return np.array([POFI_x,POFI_y,POFI_z])

def binned_statistic(x, values, func, nbins, range):

    r0, r1 = range
    
    x_mask = (x>r0) & (x<r1)
    index = np.where(x_mask)
    sampled_x = x[index]
    sampled_values = values[index]
    
    N = len(sampled_values)

    digitized = (float(nbins)/(r1 - r0)*(sampled_x - r0)).astype(int)
    S = csr_matrix((sampled_values, [digitized, np.arange(N)]), shape=(nbins, N))

    hist = [func(group) for group in np.split(S.data, S.indptr[1:-1])]
    bin_edges = np.linspace(r0, r1, nbins+1)
    
    return hist, bin_edges

def plot_segments(plot_size, digitized_loc):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # Plot the grid
    for i, j in digitized_loc:
        rect = plt.Rectangle((i, j), 1, 1, color='red')
        ax.add_patch(rect)

    # Configure the plot
    ax.set_xticks(np.arange(0, plot_size + 1, 1))
    ax.set_yticks(np.arange(0, plot_size + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
        
    # Set limits to fit the grid within the plot
    plt.grid(color='black')
    plt.xlim(0, plot_size)
    plt.ylim(0, plot_size)

    # Display the plot
    plt.show()

def mc_random_var(size, sigma, pdf='gaussian', weight=1):
     
    P = np.random.rand(np.sum(size))
    
    if np.isscalar(sigma):
        sigma = sigma * np.ones(P.shape[0])
    else:
        sigma = np.repeat(sigma, size)
        
    if np.isscalar(weight):
        weight = weight * np.ones(P.shape[0])
    else:
        weight = np.repeat(weight, size)
       
    if pdf == 'gaussian':
        return weight * sigma * np.sqrt(2) * erfinv(2*P-1)
        
    if pdf == 'uniform':
        return weight * sigma * (2*P-1)
    
    else:
        raise ValueError("Unsupported pdf type. Use 'gaussian' or 'uniform'.")
   
def explore_hdf5(group, level=0):
    """Recursively explore HDF5 groups and datasets."""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print("  " * level + f"Group: {key}")
            explore_hdf5(item, level + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * level + f"Dataset: {key} | Shape: {item.shape} | Type: {item.dtype}")
     
  #  def phi(self):
        
 #       P = np.random.rand(*self.size)
        
 #       return  P * np.pi/2
    
 #   def energy(self):