import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from .util import binned_statistic, plot_segments

class hist:
    
    def __init__(self, x, values, range=None, phi_range=None):
        self.x = x
        self.values = values
        self.range = range
        self.phi_range = phi_range
        
    def __stat_type(self, statistic):
        if statistic == 'mean':
            func = np.mean
            
        elif statistic == 'sum':
            func = np.sum
            
        elif statistic == 'count':
            def func(x):
                return x.shape[0]
            
        elif statistic == 'ones':
            def func(x):
                if x.size == 0:
                    return 0
                else:
                    return 1
            
        elif callable(statistic):
            func = statistic
        else:
            raise ValueError("statistic has to be 'mean', 'sum', 'count' or a function.")
        
        return func
        
    def segment_statistic1D(self, vbins=10, statistic='mean'):
        if self.x.ndim != 1:
            raise TypeError("Sequence to be binned has to be 1D.")
        if self.range == None:
            self.range = (self.x.min(),self.x.max())
        
        self.v_hist, self.x_bin_edges = binned_statistic(self.x, self.values, self.__stat_type(statistic), vbins, self.range)
    
    def segment_statistic2D(self, segment_size, statistic='mean'):
        if self.x.ndim != 2:
            raise TypeError("Sequence to be binned has to be 2D.")
        
        x = self.x[:,0:2]
        self.r = np.sqrt(np.einsum('...i,...i',x,x))
        
        if self.range == None:
            self.range = (self.r.min(),self.r.max())
            
        if self.phi_range == None:
            self.phi_range = (-180,180)
            
        func = self.__stat_type(statistic)
        
        r0, r1 = self.range
        phi0, phi1 = self.phi_range
        phi0 = phi0 * np.pi/180
        phi1 = phi1 * np.pi/180
        self.segment_size = segment_size
        
        if (phi0 < phi1):
            phi_mask = (np.arctan2(x[:,1],x[:,0]) > phi0) & (np.arctan2(x[:,1],x[:,0]) < phi1)
        else:
            phi_mask = (np.arctan2(x[:,1],x[:,0]) > phi0) | (np.arctan2(x[:,1],x[:,0]) < phi1)
            
        x_mask = phi_mask & (self.r > r0) & (self.r < r1)
        index = np.where(x_mask)[0]
        self.sampled_x = x[index]
        self.sampled_r = self.r[index]
        self.sampled_values = self.values[index]
        N = len(self.sampled_values)
        
        seg_max = (int(r1/segment_size) + 1) * segment_size        
        self.area_bins = int(4 * seg_max**2 / segment_size**2)
        self.segment_bins = int(2 * seg_max / segment_size)
        
        digitized_x = ((self.sampled_x[:,0] + seg_max)/segment_size).astype(int)  
        digitized_y = ((self.sampled_x[:,1] + seg_max)/segment_size).astype(int)
        self.digitized_loc = zip(digitized_x, digitized_y)
        
        self.flat_bin_indices = digitized_x * self.segment_bins + digitized_y
        S = csr_matrix((self.sampled_values, [self.flat_bin_indices, np.arange(N)]), shape=(self.area_bins, N))

        self.v_hist = [func(group) for group in np.split(S.data, S.indptr[1:-1])]
        self.x_bin_edges = np.linspace(-seg_max, seg_max, self.segment_bins + 1)
        
    def count_statistic2D(self, segment_size, v_range, v_bins = 10):
        self.v_range = v_range
        self.v_bins = v_bins
        
        def count(x):
            hist, bin_edges = binned_statistic(x, x, self.__stat_type('count'), v_bins, v_range)
            self.v_bin_edges = bin_edges
            return hist
        
        self.segment_statistic2D(segment_size, statistic = count)
        self.v_hist = np.sum(np.array(self.v_hist), axis=0)
        
    def count_segment_statistic2D(self, segment_size, v_range, v_bins = 10):
        self.v_range = v_range
        self.v_bins = v_bins
        
        def count(x):
            hist, bin_edges = binned_statistic(x, x, self.__stat_type('ones'), v_bins, v_range)
            self.v_bin_edges = bin_edges
            return hist
        
        self.segment_statistic2D(segment_size, statistic = count)
        self.v_hist = np.sum(np.array(self.v_hist), axis=0)
    
    def sps_statistic2D(self, segment_size, v_range, v_bins = 10):
        self.v_range = v_range
        self.v_bins = v_bins
        
        self.segment_statistic2D(segment_size, statistic = 'count')
        self.v_hist, self.v_bin_edges = binned_statistic(np.array(self.v_hist), np.array(self.v_hist), self.__stat_type('count'), self.v_bins, self.v_range)
    
    @property
    def x_plot(self):
        plot_segments(self.segment_bins, self.digitized_loc)
        
    @property
    def v_plot(self):
        fig = plt.figure
        plt.bar(self.v_bin_edges[:-1], self.v_hist, width=np.diff(self.v_bin_edges), edgecolor="black", align="edge")
        plt.show()
        
        
        
