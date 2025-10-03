# # Function to convert the yee grid array coordinates to the physical real coordinates 
# # Read the json file and extract the following information: resolution, cell_size, meep_unit_length, boundary layer thickness and 
# # The idea is to create an empty array of the same size as the yee grid array and then fill it with the real coordinates
# # The empty array is of the form--> np.zeros((size_x + boundary layer thickness)*res+1, (size_y + boundary layer thickness)*res+1, (size_z + boundary layer thickness)*res+1)
# # Then calculate the pixels per wavelength based on the formula: 
# # pixels_per_wavelength = resolution * meep_unit_length / wavelength
# # The real coordinates will be calculated as follows:
# # real_x = x * meep_unit_length*resolution; x is the x coordinate of the yee grid array
# # real_y = y * meep_unit_length*resolution; y is the y coordinate of the yee grid array
# # real_z = z * meep_unit_length*resolution; z is the z coordinate of the yee grid array
# # The function will return the real coordinates array with the filled real coordinates values that can be useful for analysis


# class Analysis(object):
#     '''
#     Class definining analysis tools.
#     '''
    
#     def __init__(self, sim):
#         '''
#         Defines the used sim environment : the objects and their properties as 
#         specified in the sim will be the same all throughout the analysis.

#         Arguments
#         ---------
#         sim : sim object
#             Sim to be used
#         '''
#         self.sim = sim
        
#     def image_plane_beams(self, 
#                         f = None, 
#                         wvl = None, 
#                         y_source = 0., 
#                         simres = 1,
#                         runtime = 750, 
#                         beam_w0 = 10,
#                         plot_amp = False, 
#                         saveh5 = False, 
#                         filename = 'test',
#                         parallel = False):
#         '''
#         Sends gaussian beams from the image plane and recovers 
#         the amplitudes at the aperture. 

#         Arguments
#         ---------
#         f : float or list, optional
#             Frequency of the source (default : None)
#         wvl : float or list, optional
#             Wavelength of the source (default : None)
#         y_source : float or list, optional
#             Position of source on image plane along y axis. (default : 0)
#         simres : float or list, optional
#             Resolution of simulation (default : 1)
#         runtime : float, optional
#             Runtime of simulation. Should roughly be 
#             system size along optical axis. (default : 750)
#         beam_w0 : float or list, optional
#             Size of beam waist of gaussian source (default : 10)
#         plot_amp : bool, optional
#             Whether to plot the amplitude of field at aperture. 
#             (default : False)
#         saveh5 : bool, optional
#             Whether to save the amplitudes in an h5 file. (default : False)
#         filename : list of str, optional
#             Names of the files to be saved
#         parallel : bool, optional
#             Whether the code is running in parallel
#         Notes
#         -----
#         If the arguments above are given as lists, they should all be 
#         the same length, and a sim will be run for each element, taking
#         the property i of each list.
#         '''

#         if not isinstance(wvl, list) :
#             wvl = [wvl]
#             y_source = [y_source]
#             beam_w0 = [beam_w0]
#             filename = [filename]

#         # Adaptation to specify either in wavelength or frequency :
#         if wvl is not None :
#            f = [1/wvl[k] for k in range(len(wvl))]

#         if f is not None:
#             wvl = [1/x for x in f]

#         self.wvl = wvl

#         Nb_src = len(wvl)

#         #Initialize the electric fields list
#         self.list_efields = [[] for k in range(Nb_src)]

#         #Iterates over the number of sources
#         for k in range(Nb_src):

#             #Defines the source at the appropriate height on the image plane
#             self.sim.define_source(wvl = wvl[k], 
#                                    sourcetype = 'Gaussian beam', 
#                                    y = y_source[k], 
#                                    size_x = 0, 
#                                    size_y = self.sim.OS.size_y, 
#                                    beam_width = beam_w0[k])

            
#             #Runs the sim
#             self.sim.run_sim(runtime, simres = simres, ff_angle = 80, ff_npts = 800)

#             #self.sim.plot_efield()

#             self.sim.get_MEEP_ff(saveplot = False,
#                     parallel = parallel,
#                     saveh5 = True,
#                     filename = filename[k] + '_n2far',
#                     ylim = -40)

#             #Gets the complex electric field and adds it to the plot
#             E_field = self.sim.get_complex_field(plot_amp = plot_amp,
#                                     saveh5 = saveh5, 
#                                     filename = filename[k] + '_{}'.format(int(k)),
#                                     parallel = parallel)

#             #Updates the list of fields
#             self.list_efields[k] = E_field

#     def beam_FT(self, 
#                 zero_pad = 15,
#                 savebeam = False,
#                 parallel = False,
#                 filename = None):

#         '''
#         Gets the Fourier Transforms of the complex electric fields at aperture.

#         Arguments
#         ---------
#         zero_pad : float, optional
#             Multiplicative factor to the length of the field list,
#             which is padded with zeros in the added length.

#         Returns
#         -------
#         freq : array
#             List of the frequencies at which the FFT has been done
#         FFTs : list of arrays
#             Each array contains the FFT for the k-th source.
#         '''

#         #Initialize the list
#         FFTs = [[] for k in range(len(self.list_efields))]

#         res = self.sim.simres

#         #List of frequencies
#         freq = np.fft.fftfreq(len(self.list_efields[0])*zero_pad, d = 1/res)

#         #Iterate over the number of sources
#         for k in range(len(self.list_efields)):

#             #FFT over the field
#             fft = np.fft.fft(self.list_efields[k], 
#                     n = zero_pad*len(self.list_efields[k]))

#             #FFT is normalized by its max
#             FFTs[k] = np.abs(fft) 
#             FFTs[k] = FFTs[k]/np.max(FFTs[k])

#         if savebeam :
#             if parallel :
#                 h = h5py.File(filename + '.h5', 'w', 
#                             driver ='mpio', 
#                             comm=MPI.COMM_WORLD)
#             else: 
#                 h = h5py.File(filename + '.h5', 'w')
#             h.create_dataset('freq', data=freq, dtype = 'float64')
#             h.create_dataset('beams', data=FFTs, dtype = 'float64')
#             aper = self.sim.OS.aper_size
#             h.create_dataset('aper_size', data = aper, dtype = 'float64')
#             h.close()
#         return freq, FFTs

#     def open_saved_beams(self, filename, parallel = False):

#         name = filename + '.h5'
#         if parallel :
#             data = h5py.File(name, 'r', driver ='mpio', 
#                             comm=MPI.COMM_WORLD)
#         else : 
#             data = h5py.File(name, 'r')
#         aper = data['aper_size']
#         FFTs = data['beams']
#         freq = data['freq']

#         beam = np.copy(FFTs)
#         freq_copy = np.copy(freq)
#         self.sim.OS.aper_size = np.copy(aper)
#         data.close()
#         return freq_copy, beam


#     def plotting(self, fftfreq, FFTs, wvl,
#                 deg_range = 20,
#                 ylim = -60, 
#                 symmetric_beam = True,
#                 legend = None,
#                 print_solid_angle = False,
#                 print_fwhm = False,
#                 savefig = False,
#                 path_name = 'plots/meep_guide_plot',
#                 seq_col = False):
#         '''
#         Plots far field beam

#         Arguments
#         ---------
#         fftfreq : float
#             Array of the frequencies of the FFT
#         FFTS : float
#             List of the normalized beams of the FFT
#         wvl : float or list of floats
#             Wavelengths of the beams 
#         deg_range : float, optional
#             Range in degrees of the plot (default : 20)
#         ylim : float, optional
#             Min amplitude of the plot, in dB (default : -60)
#         symmetric_beam : bool, optional
#             If the beam is symmetric, if true only plots half of the beam
#             (default : True)
#         legend : list of str, optional
#             Legend of the various far fields plotted (default : None)
#         print_solid_angle : bool, optional
#             Whether to print the solid angle (default : False)
#         print_fwhm : bool, optional
#             Whether to print the best fit gaussian FWHM (default : False)
#         savefig : bool, optional
#             Whether to save the figure (default : False)
#         path_name : str, optional
#             Path and name of the plot to be saved 
#             (default : 'plots/meep_guide_plot')
#         seq_col : bool, optional
#             Whether to set a sequential colormap (default : False)
#         '''

#         deg = np.arctan(fftfreq*wvl)*180/np.pi
#         rads = np.array(deg) * np.pi/180

#         col = plt.cm.jet(np.linspace(0,1,len(FFTs))) 
        

#         plt.figure(figsize = (8,6))
        
#         def gaussian(x, stddev, mean):
#             return np.exp(-(((x-mean)/4/stddev)**2))
        
#         for k in range(len(FFTs)):

#             fft_k = (FFTs[k] / (np.cos(rads)**2))**2
#             fft_k = fft_k / np.max(fft_k)
#             fft_dB = 10*np.log10(fft_k)
#             middle = int(len(fft_k)/2)

#             #BEAM SOLID ANGLE CALCULATION
#             if print_solid_angle :
                
                
#                 x_span = np.append(rads, 0)
#                 integrand = np.append(fft_k, fft_k[0])
#                 integrand *= np.sin(x_span)
#                 right_part = np.trapz(integrand[:middle], x = x_span[:middle])
#                 #left_part = np.trapz(integrand[middle:], x = x_span[middle:])
#                 solid_angle = right_part #+ left_part
#                 print('Beam n.{} solid angle : {:.3e} srads'.format(k, 
#                     solid_angle*2*np.pi))
            
#             if legend is not None : 
#                 plt.plot(deg[:middle], fft_dB[:middle], 
#                     label = '{}'.format(legend[k]), color = col[k])

#             if legend is None :

#                 plt.plot(deg[:middle], fft_dB[:middle], color = col[k])
#                 #TESTING, ignore this
#                 #plt.plot(self.sim.angles, self.sim.ffmeep)

#             #BEST FIT GAUSSIAN FWHM
#             if print_fwhm :

#                 #Fit is done around the gaussian portion of the beam
#                 maxidx = np.argmax(fft_k)
#                 if maxidx == len(fft_k) - 1:
#                     maxidx = 0
#                 i = 0
#                 while fft_k[maxidx + i] > fft_k[maxidx + i + 1] :
#                     i += 1

#                 xdata = deg[maxidx - i : maxidx + i ]
#                 ydata = fft_k[maxidx - i : maxidx + i ]
#                 if maxidx - i <= 0:
#                     xdata = np.concatenate((deg[maxidx - i:], deg[:maxidx + i]))
#                     ydata = np.concatenate((fft_k[maxidx - i:], fft_k[:maxidx + i]))
#                 p0 = [1,1]
#                 if maxidx <=10:
#                     p0 = [1,0]
#                 popt, psig = sc.curve_fit(gaussian, xdata, ydata, p0 = p0)
#                 fwhm = np.abs(4*popt[0]*np.sqrt(np.log(2)))
#                 fwhm_th = wvl/self.sim.OS.aper_size*180/np.pi
#                 print('Best fit Gaussian FWHM : {:.2f}deg'.format(2*fwhm))
#                 print('Theoretical FWHM : {:.2f}deg'.format(fwhm_th))
#                 gauss = gaussian(deg[:middle], popt[0], popt[1]) + 1e-10
#                 y = 10*np.log10(gauss)

#                 plt.plot(deg[:middle], y, linestyle = '--', color = col[k])
#                                         #color = 'C{}'.format(int(k)))

#         plt.ylim((ylim, 0))
#         plt.xlabel('Angle [deg]', fontsize = 14)
#         plt.ylabel('Power [dB]', fontsize = 14)
#         plt.xticks(fontsize = 12)
#         plt.yticks(fontsize = 12)

#         if symmetric_beam :
#             plt.xlim((0,deg_range))
#         if not symmetric_beam : 
#             plt.xlim((-deg_range, deg_range))

#         if legend is not None :
#             plt.legend(loc = 'upper right', fontsize = 12)

#         #Additional plotting tools
#         """
#         fwhm = args.wvl*0.28648

#         plt.vlines([-fwhm/2, fwhm/2], -100, 0, color = 'grey', linestyle = 'dashdot')
#         plt.vlines([fwhm_fft[0]/2], -100, 0, color='grey', linestyle = '--', alpha = 0.7)
#         plt.annotate('Expected FWHM : {:.2f}deg'.format(fwhm), 
#             xy = (.25, .9), xycoords='figure fraction', color = 'grey')
#         plt.annotate('Beam FWHM : {:.2f}deg'.format(fwhm_fft[0]), 
#             xy = (.25, .87), xycoords='figure fraction', color = 'grey', alpha = 0.7)
#         """

#         #plt.annotate('Field FWHM : {:.2f}mm'.format(fwhm_ap[0]), 
#         #    xy = (.1, .84), xycoords='figure fraction')
#         plt.tight_layout()
#         plt.show()
#         if savefig :
#             plt.savefig('{}.png'.format(path_name))
#         plt.close()
