import numpy as np
import json
import ParameterContainer

class global_parameters(ParameterContainer.ParameterContainer):
    """
    The parameter class storing the simulation parameters
    is derived from the ParameterContainer class.

    Parameters used (mostly) by different classes should be seperated into
    different functions.
    Common parameters are set in the set_default_params function
    """

    def __init__(self, params_fn=None):#, output_dir=None):
        """
        Keyword arguments:
        params_fn -- string, if None: set_filenames and set_default_params will be called
        """

        if params_fn == None:
            self.params = {}
            self.set_default_params()
            # ... set other parameters in a structured way
        else:
            self.params = self.load_params_from_file(params_fn)

        # will set the filenames
        super(global_parameters, self).__init__() # call the constructor of the super/mother class


    def set_default_params(self):
        """
        Here all the simulation parameters NOT being filenames are set.
        """

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['t_sim'] = 300.     # simulation time in [ms]
        self.params['dt'] = 0.1         # simulation time step
        self.params['dt_volt'] = 1 * self.params['dt'] # time step for voltage recording (can be integer multiples of dt)
        self.params['n_mc_per_hc'] = 3          # number of minicolumns per hypercolumn
        self.params['n_hc'] = 1                 # number of hypercolumns
        self.params['n_mc'] = self.params['n_mc_per_hc'] * self.params['n_hc'] # total number of minicolumns in the network
        self.params['n_exc_per_mc'] = 40         # number of neurons per minicolumn
        self.params['n_inh_per_hc'] = 10

        # ############################################################
        # C O N N E C T I V I T Y    P A R A M E T E R S
        # ############################################################
        self.params['w_input_exc'] = 5.     # [nS] how strong the input is connected to the cell

        # exc - exc within one minicolumn
        self.params['p_ee_local'] = .5
        self.params['w_ee_local'] = 5.
        self.params['delay_ee_local'] = 5.

        # exc - exc: global connectivity between minicolumns in different hypercolumns
        self.params['w_ee_global_max'] = 4.
        self.params['w_ei_global_max'] = 15.
        self.params['delay_ee_global'] = 1. # [ms]

        # unspecific  /  normalizing  feedback inhibition
        # exc - inh:
        self.params['w_ei_unspec'] = 5.     # untrained, unspecific PYR -> Basket cell connections
        self.params['p_ei_unspec'] = .75    # probability for PYR -> Basket connections
        self.params['delay_ei_unspec'] = 1.

        # inh - exc:
        self.params['w_ie_unspec'] = -200.  # untrained, unspecific Basket -> PYR connections
        self.params['p_ie_unspec'] = .7     # probability for Basket -> PYR Basket connections
        self.params['delay_ie_unspec'] = 1.

        # Specific inhibition
        self.params['w_ie_spec'] = -50.     # RSNP -> PYR, effective only after training
        self.params['p_ie_spec'] = 1.       # RSNP -> PYR
        self.params['delay_ie_spec'] = 1.

        # inh - inh
        self.params['w_ii_unspec'] = 1. # untrained, unspecific Basket -> PYR connections
        self.params['p_ii_unspec'] = .7 # probability for Basket -> PYR Basket connections
        self.params['delay_ii_unspec'] = 1.


        # ##########################################
        # N E U R O N A L     P A R A M E T E R S
        # ##########################################
        self.params['g_L'] = 16.6667        # [nS] leakage conductance, influences the integration time constant of the neuron membrane (tau_mem)
        self.params['neuron_type'] = 'iaf_cond_exp'




    def set_filenames(self, folder_name=None):
        """
        This funcion is called if no params_fn is passed
        """

        self.set_folder_names(folder_name)

        # Since data_path is already set to spiketimes_folder --> files will be in the spiketimes_folder
        self.params['exc_volt_fn'] = 'exc_volt_'
        self.params['exc_spikes_fn'] = 'exc_spikes_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['exc_spikes_fn_merged'] = 'exc_merged_spikes.dat' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['inh_volt_fn'] = 'inh_volt_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['inh_spikes_fn'] = 'inh_spikes_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['inh_spikes_fn_merged'] = 'inh_merged_spikes.dat' # data_path is already set to spiketimes_folder --> files will be in this subfolder

        self.params['gids_fn'] = self.params['parameters_folder'] + 'cell_gids.json'

        # input spike files
        self.params['input_st_fn'] = self.params['input_folder'] + 'input_spikes_'

    def set_folder_names(self, folder_name=None):

        if folder_name == None:
            folder_name = 'Output/'
        # else: it is a likely a parameter sweep and the folder name is set from 'outside'
        assert(folder_name[-1] == '/'), 'ERROR: folder_name must end with a / '

        self.params['folder_name'] = folder_name
        self.params['parameters_folder'] = '%sParameters/' % self.params['folder_name']
        self.params['figures_folder'] = '%sFigures/' % self.params['folder_name']
        self.params['connections_folder'] = '%sConnections/' % self.params['folder_name']
        self.params['data_folder'] = '%sData/' % (self.params['folder_name']) # for storage of analysis results etc
        self.params['input_folder'] = '%sInputSpikes/' % (self.params['folder_name'])
        self.params['spiketimes_folder'] = '%sSpikes/' % self.params['folder_name']
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['parameters_folder'], \
                            self.params['figures_folder'], \
                            self.params['connections_folder'], \
                            self.params['data_folder'], \
                            self.params['spiketimes_folder'], \
                            self.params['input_folder'] ]
        self.params['params_fn_json'] = '%ssimulation_parameters.json' % (self.params['parameters_folder'])


    def load_params_from_file(self, fn):
        """
        Loads the file via json from a filename
        Returns the simulation parameters stored in a file
        Keyword arguments:
        fn -- file name
        """
        f = file(fn, 'r')
        print 'Loading parameters from', fn
        self.params = json.load(f)
        return self.params


