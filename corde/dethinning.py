import numpy as np
import matplotlib.pyplot as plt
import h5py
from corsikaio import CorsikaParticleFile, as_dict
from .util import normalize, cosine, rotate, dmax, mc_random_var
from .histogram import hist
            
class corsika_dethin:
    
    def __init__(self, file_dir, file_list, storage=None):
        self.__has_run = False
        self.file_dir = file_dir
        self.file_list = file_list
        self.storage = storage
        
        if self.storage == 'HDF5':
            self.h5file = h5py.File("events.h5", "w")
        elif storage == None:
            pass
        else:
            raise ValueError("storage type not supported.")

        self.events = []
        self.pid = []
        self.del_t = []
        self.xi = []
        self.pi = []
        self.thinning_weight = []
        self.z = []
        self.theta = []
        self.phi = []
        self.dmax_list = []
        self.regen_xi = []
        self.regen_pid = []
        
        event_num = 0
        
        for file in self._files:
            for event in file:
                event_num += 1
                self._generate_data(event)
                self._regenerate
                if self.storage == 'HDF5':
                    self._store_hdf5(event_num)
                    self._delete_data
                    
        self.__has_run = True
    
    @staticmethod
    def private(f):
        def wrapper(self, *args, **kwargs):
            if not self.__has_run:
                return f(self, *args, **kwargs)
            else:
                raise ValueError("This function is private and has already been run.")
        return wrapper
    
    @property
    @private
    def _files(self):
        file = []
        for f in self.file_list:
            file.append(CorsikaParticleFile(self.file_dir+f,thinning=True))
        return file
    
    @private
    def _store_hdf5(self, event_num):
        event_group = self.h5file.create_group(f"event{event_num}")
        event_group.create_dataset("pid", data=self.pid[-1])
        event_group.create_dataset("del_t", data=self.del_t[-1])
        event_group.create_dataset("xi", data=self.xi[-1])
        event_group.create_dataset("pi", data=self.pi[-1])
        event_group.create_dataset("thinning_weight", data=self.thinning_weight[-1])
        event_group.create_dataset("z", data=self.z[-1])
        event_group.create_dataset("zenith", data=self.theta[-1])
        event_group.create_dataset("azimuth", data=self.phi[-1])
        event_group.create_dataset("dmax_list", data=self.dmax_list[-1])
        event_group.create_dataset("regen_xi", data=self.regen_xi[-1])
        event_group.create_dataset("regen_pid", data=self.regen_pid[-1])
    
    @property
    @private
    def _delete_data(self):
        self.pid.clear()
        self.del_t.clear()
        self.xi.clear()
        self.pi.clear()
        self.thinning_weight.clear()
        self.z.clear()
        self.theta.clear()
        self.phi.clear()
        self.dmax_list.clear()
        self.events.clear()
        self.regen_xi.clear()
        self.regen_pid.clear()
    
    @private
    def _generate_data(self, event):
        self.pid.append((event.particles['particle_description']/1000).astype(int))
        self.del_t.append(event.particles['t'])
        self.xi.append(np.array([event.particles['x'],
                                 event.particles['y'],
                                 np.full(event.particles['x'].shape,event.header['observation_height'][0])]).T)
        self.pi.append(np.array([event.particles['px'],event.particles['py'],-event.particles['pz']]).T)
        self.thinning_weight.append(event.particles['thinning_weight'])
        self.z.append(np.abs(event.header['first_interaction_height']))
        self.theta.append(event.header['zenith'])
        self.phi.append(event.header['azimuth'])
        self.dmax_list.append(dmax(self.del_t[-1],self.xi[-1],self.pi[-1],self.z[-1],self.theta[-1],self.phi[-1]))
        self.events.append(event)
    
    @property
    @private
    def _non_zero_samples(self):
        particle_num = (np.ceil(self.thinning_weight[-1]) - 1).astype('int')
        mask = (particle_num > 0)
        index = np.where(mask)[0]
        return particle_num[index], index
    
    @property
    @private
    def _theta_sigma(self):
        mask = (self.pid[-1] > 0) & (self.pid[-1] < 4)
        return np.where(mask, 3*np.pi/180*10**(-5), 1*np.pi/180*10**(-5))
        
    @private
    def _vertex_location(self, pnorm, index):
        cos_alpha = cosine(np.array([0,0,-1]), pnorm)
        return self.xi[-1][index,:] - (self.dmax_list[-1][index]/cos_alpha).reshape(pnorm.shape[0], 1) * pnorm
        
    @property
    @private
    def _regenerate(self):
        
        particle_num, index = self._non_zero_samples
        
        lat_x = self.xi[-1][index,0:2]
        lat_r = np.squeeze(np.sqrt(np.einsum('...i,...i',lat_x,lat_x)))
        
        theta = mc_random_var(particle_num, self._theta_sigma[index], weight=lat_r)
        phi = mc_random_var(particle_num, np.pi/2, pdf='uniform', weight=lat_r)
        
        pnorm_temp = normalize(self.pi[-1][index,:])
        pnorm = np.repeat(pnorm_temp, particle_num, axis=0)
        dmax_temp = np.repeat(self.dmax_list[-1][index], particle_num, axis=0)
        
        w = np.cross(np.array([0,0,-1]),pnorm)
        new_pnorm = rotate(rotate(pnorm,w,theta),pnorm,phi)
        ver_loc = np.repeat(
            self._vertex_location(pnorm_temp, index), particle_num, axis=0)
        cos_alpha = cosine(np.array([0,0,-1]), new_pnorm)
        
        self.regen_xi.append(ver_loc + (dmax_temp/cos_alpha).reshape(new_pnorm.shape[0],1) * new_pnorm)
        self.regen_pid.append(np.repeat(self.pid[-1][index],particle_num, axis=0))
        
    def hist_plot(self, event_num=0, type='e', seg_size=600, x_range=[0,10000], phi_range=None, v_range=[0,100], v_bins=10):
        
        pid = np.concatenate((self.pid[event_num], self.regen_pid[event_num]), axis=0)
        xi = np.concatenate((self.xi[event_num], self.regen_xi[event_num]), axis=0)

        if type=='e':
            mask = (pid > 1) & (pid < 4)
            
        index = np.where(mask)[0]
        pid = pid[index]
        xi = xi[index]
        hist_ob = hist(xi,np.ones(pid.shape), range=x_range, phi_range=phi_range)
        hist_ob.sps_statistic2D(seg_size, v_range, v_bins)
        binned_hist = hist_ob.v_hist
        
        fig = plt.figure
        plt.bar(hist_ob.v_bin_edges[:-1], binned_hist, width=np.diff(hist_ob.v_bin_edges), edgecolor="black", align="edge")
        plt.show()
        
    def hist_plot_all(self, type='e', seg_size=600, x_range=[0,10000], phi_range=None, v_range=[0,100], v_bins=10):
        
        binned_list = []
        
        for event_num in range(len(self.pid)):
            pid = np.concatenate((self.pid[event_num], self.regen_pid[event_num]), axis=0)
            xi = np.concatenate((self.xi[event_num], self.regen_xi[event_num]), axis=0)

            if type=='e':
                mask = (pid > 1) & (pid < 4)
            
            index = np.where(mask)[0]
            pid = pid[index]
            xi = xi[index]
            hist_ob = hist(xi,np.ones(pid.shape), range=x_range, phi_range=phi_range)
            hist_ob.sps_statistic2D(seg_size, v_range, v_bins)
            binned_list.append(hist_ob.v_hist)
        
        binned_hist = np.sum(np.array(binned_list),axis=0)
        fig = plt.figure
        plt.bar(hist_ob.v_bin_edges[:-1], binned_hist, width=np.diff(hist_ob.v_bin_edges), edgecolor="black", align="edge")
        plt.show()
       
            

    