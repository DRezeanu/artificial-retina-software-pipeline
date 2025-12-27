'''
Module for reading in spike sorting outputs from manual analysis (elecResp), 
and automated analysis (Autosort, elecRespAuto).

NOTE: these functions assume SINGLE electrode stimulation. There might be some
assumptions made between the overloaded "pattern" field, the meaning of which 
changes when multiple stimulating electrodes are used. In single electrode
stimulation, pattern and stimulating electrode are identical, for convienience.

@author Alex Gogliettino
@date 2020-05-09
'''

import numpy as np
import scipy as sp
from scipy.io import loadmat
import os
import electrode_map as elcmp
import visionloader as vl
import pdb

# Constants.
ELECRESPAUTO_PREFIX = 'elecRespAuto_p'
MAT_SUFFIX = '.mat'
GLOBALS_FILE_EXT = 'globals'
SPIKES_CORRECTED_FIELD = 'spikesCorrected'
CONFIDENCE_FIELD = 'correctionConfidence'
CELL_CORRECTED_FIELD = 'isCellCorrected'

class ElecRespAutoReader():
    '''
    Class for reading elecRespAuto structs written by Autosort. There might be
    some compatibility issues if using older elecRespAuto files, written from
    previous versions of Autosort (i.e. before spike_sorting_GM3), as the 
    format changed slightly. Some fields may not be present.

    Example usage:
        import elecresploader as el
        autosortpath = '/Volumes/Analysis/2020-02-27-2/'\
                       'Autosort-single-agogliet-multi-doPostBundle/'
        pattern_no = 500
        er = el.ElecRespAutoReader(autosortpath,pattern_no)
        artifacts = er.get_artifacts()

        ...
        
        See source code for other methods.
    '''

    def __init__(self,autosortpath: str, pattern_no: int) -> None:

        '''
        Constructor, initializes object and loads data. Sets the file flag
        accordingly, returns if file doesn't exist.
        '''
        assert os.path.isdir(autosortpath),"%s not a path"%autosortpath
        self.elecrespauto_path = os.path.join(autosortpath,
                                              ELECRESPAUTO_PREFIX + 
                                              str(pattern_no) + 
                                              MAT_SUFFIX)

        if not os.path.isfile(self.elecrespauto_path):
            '''
            print('elecRespAuto struct for pattern %s not written'
                  %str(pattern_no))
            return None
            '''
            self.isfile = False
            return

        self.isfile = True
        self.elecrespauto = loadmat(self.elecrespauto_path,
                                    struct_as_record=False,
                                    verify_compressed_data_integrity=False)

    def get_elecrespauto_dict(self) -> dict:
        '''
        Args:
            N/A

        Returns:
            Dictionary with several elecRespAuto fields and their values.

        Packs all the data into a dictionary.
        '''

        # Get major fields from the struct. 
        elecrespauto = self.elecrespauto['elecRespAuto']
        stiminfo = elecrespauto[0,0].stimInfo
        neuroninfo = elecrespauto[0,0].neuronInfo
        converg_log = elecrespauto[0,0].Log
        bundle = elecrespauto[0,0].bundle

        # Load subfields with general stimulation information.
        amplitudes = stiminfo[0,0].listAmps
        connected_elecs = stiminfo[0,0].ActiveElectrodes
        breakpoints = stiminfo[0,0].breakpoints
        numtrials = stiminfo[0,0].nTrials
        stim_elecs = stiminfo[0,0].stimElecs

        # Sometimes, artifacts are not written (user option in wrapper). 
        try:
            artifacts = stiminfo[0,0].Arts
        except AttributeError: 
            artifacts = None

        bundle_ind = bundle[0,0].onsetC
        bundle_thresh = (bundle[0,0].onset * -1) # To be negative current.

        ''' Load subfields with cell-specific information. Everything is 
        organized by cell (example: third cell in sorted-cells will have 
        third indexed templates.)
        '''
        templates = neuroninfo[0,0].templates
        sorting_elecs = neuroninfo[0,0].ActiveElectrodes
        sorted_cells = neuroninfo[0,0].neuronIds
        sorted_spikes = neuroninfo[0,0].spikes

        # Iterations are pattern specific, NOT cell based.
        iterations = converg_log[0,0].Iter

        # Load subfields, if exist, of the PCA template matching corrections.
        neuron_fields = neuroninfo[0,0]._fieldnames

        if SPIKES_CORRECTED_FIELD in neuron_fields:
            spikes_corrected = getattr(neuroninfo[0,0],SPIKES_CORRECTED_FIELD)
        else:
            spikes_corrected = None

        if CELL_CORRECTED_FIELD in neuron_fields:
            cell_corrected = getattr(neuroninfo[0,0],CELL_CORRECTED_FIELD)
        else:
            cell_corrected = None

        if CONFIDENCE_FIELD in neuron_fields:
            confidence = getattr(neuroninfo[0,0],CONFIDENCE_FIELD)
        else:
            confidence = None

        # Write the arrays to the dictionary. 
        elecrespauto_dict = dict()
        elecrespauto_dict['amplitudes'] = amplitudes
        elecrespauto_dict['connected_elecs'] = connected_elecs
        elecrespauto_dict['breakpoints'] = breakpoints
        elecrespauto_dict['numtrials'] = numtrials
        elecrespauto_dict['artifacts'] = artifacts
        elecrespauto_dict['templates'] = templates
        elecrespauto_dict['sorting_elecs'] = sorting_elecs
        elecrespauto_dict['sorted_cells'] = sorted_cells
        elecrespauto_dict['iterations'] = iterations
        elecrespauto_dict['bundle_ind'] = bundle_ind
        elecrespauto_dict['sorted_spikes'] = sorted_spikes
        elecrespauto_dict['bundle_thresh'] = bundle_thresh
        elecrespauto_dict['stim_elecs'] = stim_elecs
        elecrespauto_dict['spikes_corrected'] = spikes_corrected
        elecrespauto_dict['cell_corrected'] = cell_corrected
        elecrespauto_dict['confidence'] = confidence
        return elecrespauto_dict

    def get_stim_elecs(self) -> int or np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Stimulating electrode(s).
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        return elecrespauto_dict['stim_elecs'].flatten()

    def get_amplitudes(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Array of amplitudes.

        Gets the amplitudes from the experiment, in µA. Note: these have
        been made POSITIVE, so they are absolute values.
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        return elecrespauto_dict['amplitudes'].flatten()

    def get_num_trials(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Array of trial numbers at each amplitude.
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        return elecrespauto_dict['numtrials'].flatten()

    def get_electrode_map(self) -> tuple:
        '''
        Args:
            N/A

        Returns:
            Array ID from the Globals file.

        Gets the electrode map and disconnected electrodes. Wrapper around
        GlobalsFileReader.
        '''

        # Extract the white noise run path.
        elecrespauto = self.elecrespauto['elecRespAuto']
        eipath = elecrespauto[0,0].path[0,0].eiFilePath[0]

        # Set path and datarun, and get the array ID.
        analysis_path = os.path.dirname(eipath)
        datarun = os.path.split(analysis_path)[1]

        with vl.GlobalsFileReader(analysis_path,
                                  datarun,
                                  globals_extension=GLOBALS_FILE_EXT) as gbfr:
            return gbfr.get_electrode_map()

    def get_connected_electrodes(self) -> np.ndarray:
        '''
        Args
            N/A

        Returns:
            Array of the connected electrodes.
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        return elecrespauto_dict['connected_elecs'].flatten()

    def get_artifacts(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Array of artifacts for each electrode, found by autosort.

        Given the 519 array having only 512 active electrodes, for whatever 
        reason, Autosort returns the artifacts as a 512 x t matrix, which 
        makes indexing a nightmare. So, if the data come form the 519 array, 
        this function artificially inserts zeros into the artifacts tensor
        to avoid confusion down the line.
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        artifacts = elecrespauto_dict['artifacts']
        electrode_map,disconnected_electrodes = self.get_electrode_map()

        # If there are no disconnected electrodes.
        if not disconnected_electrodes:
            return artifacts

        # Otherwise, pad the tensor with zeros to fill up to expected size. 
        num_elecs = electrode_map.shape[0]
        artifacts_pad = np.zeros((artifacts.shape[0],
                                 num_elecs,
                                 artifacts.shape[2]))
        connected_electrodes = self.get_connected_electrodes()
        conn_elec_inds = connected_electrodes - 1
        artifacts_pad[:,conn_elec_inds,:] = artifacts
        return artifacts_pad

    def get_iterations(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Array of the iterations until convergence log.

        Returns the amplitude indexed iterations until convergence log.
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        return elecrespauto_dict['iterations'].flatten()

    def get_breakpoints(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Array of the breakpoints (1-indexed)

        The Litke stimulation system has different 'breakpoints' where the 
        amplitude range changes (because of DAC precision), and changes the
        artifact shape. These indices are returned.

        These are 1-indexed, but 0 I think denotes 'the amplitude before 
        the first amplitude'.
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        return elecrespauto_dict['breakpoints'][0][0].flatten()

    def get_cells(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Array of the cells found by autosort.
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        return elecrespauto_dict['sorted_cells'][0]

    def get_cell_index(self, cell: int) -> np.ndarray or None:
        '''
        Args:
            cell ID of interest (int)

        Returns:
            Index of the sorted cell. If cell wasn't sorted, returns None.

        Note: this assumes that each cell appears exactly once. There have
        been reports of duplicate cell ids in this field (see Raman Vilkhu),
        so be on the lookout for that.
        '''
        cells = list(self.get_cells())

        if cell not in cells:
            return None

        return cells.index(cell)

    def is_cell_analyzed(self, cell: int) -> bool:
        '''
        Args:
            cell ID of interest.

        Returns:
            Boolean indicating whether the cell was analyzed by autosort (true)
            false otherwise.
        '''
        if self.get_cell_index(cell) is None:
            return False

        return True

    def is_cell_corrected(self,cell: int) -> bool:
        '''
        Args:
            cell ID of interest

        Returns:
            Boolean indicating whether the cell was corrected by the PCA 
            template matching corrections algorithm
        '''

        elecrespauto_dict = self.get_elecrespauto_dict()
        cell_corrected = elecrespauto_dict['cell_corrected']

        if cell_corrected is None or not self.is_cell_analyzed(cell):
            return False

        # Get cell index, and figure out if corrected.
        cell_ind = self.get_cell_index(cell)

        if cell_corrected[0][cell_ind] == 0:
            return False

        return True

    def get_corrected_spikes_by_cell(self, cell: int) -> np.ndarray or None:
        '''
        Args:
            cell ID of interest (int)

        Returns:
            Matrix (number of amplitudes by trials) corresponding to the
            found spike latencies by autosort, or None if the cell wasn't 
            sorted, or if the corrections weren't applied to this pattern.
            These are the corrected spikes from the PCA template 
            matching Autosort corrections.
        '''

        elecrespauto_dict = self.get_elecrespauto_dict()
        spikes_corrected = elecrespauto_dict['spikes_corrected']

        if spikes_corrected is None or not self.is_cell_corrected(cell):
            return None

        # Get corrected spikes, indexing depending on number of cells.
        num_cells = self.get_cells().shape[0]
        cell_ind = self.get_cell_index(cell)

        if num_cells == 1:
            return spikes_corrected

        return spikes_corrected[0,cell_ind]

    def get_correction_confidence_by_cell(self, cell: int) -> np.ndarray or None:
        '''
        Args:
            cell ID of interest (int)

        Returns:
            Matrix (number of amplitudes by trials) corresponding to whether
            the correction algorithm was confident in the result (1), or not (0).
            Returns None if the pattern wasn't corrected, or the cell wasn't
            analyzed/corrected.
        '''

        elecrespauto_dict = self.get_elecrespauto_dict()
        confidence = elecrespauto_dict['confidence']

        if confidence is None or not self.is_cell_corrected(cell):
            return None

        # Handle edge case of single vs. multiple cells, accordingly.
        num_cells = self.get_cells().shape[0]
        cell_ind = self.get_cell_index(cell)

        if num_cells == 1:
            return confidence

        return confidence[cell_ind,:,:]

    def get_spikes_by_cell(self, cell: int) -> np.ndarray or None:
        '''
        Args:
            cell ID of interest (int)

        Returns:
            Matrix (number of amplitudes by trials) corresponding to the
            found spike latencies by autosort, or None if the cell wasn't 
            sorted
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        spikes_all = elecrespauto_dict['sorted_spikes']
        cell_ind = self.get_cell_index(cell)

        if cell_ind is None: 
            return None

        # Handle edge case of only one cell in struct.
        if self.get_cells().shape[0] == 1:
            return spikes_all

        return spikes_all[0][cell_ind]

    def get_sorting_electrodes_by_cell(self, cell: int) -> np.ndarray or None:
        '''
        Args:
            cell ID of interest (int)

        Returns:
            Array of sorting electrodes, or None if the cell wasn't sorted.

        Because again of the 519 array problem, and how Autosort deals with
        the disconnected electrodes, this function will transform the values
        written to the elecRespAuto struct into TRUE electrode IDs, to avoid
        confusion. Basically, in the case of disconnected electrodes, written
        to the struct are the connected electrode indices, and so we must 
        transform the connected indices to true electrode values.
        '''
        elecrespauto_dict = self.get_elecrespauto_dict()
        sorting_electrodes_all = elecrespauto_dict['sorting_elecs']
        cell_ind = self.get_cell_index(cell)

        if cell_ind is None:
            return None

        # If no disconnected electrodes, we can return this as is.
        electrode_map,disconnected_electrodes = self.get_electrode_map()

        if not disconnected_electrodes:
            return sorting_electrodes_all[0][cell_ind].flatten()

        # Otherwise, we have to convert from connected indices to electrodes.
        connected_electrodes = self.get_connected_electrodes()
        connected_indices = sorting_electrodes_all[0][cell_ind].flatten() - 1
        return connected_electrodes[connected_indices]

    def __enter__(self):
        return self

    def __exit__(self,exc_type, exc_value,traceback):
        del self.elecrespauto
        return

    def close(self):
        del self.elecrespauto
        return


class ElecRespReader():
    '''
    Class for reading elecResp structs from manual spike sorting analysis.
    
    The most up-to-date naming convention has each file named 
    elecResp_nXXX_pYYY_rZZZ.mat', where XXX is the neuron, YYY is the 
    stimulating electrode, and ZZZ is the center recording electrode
    used during human inspection. Older naming conventions may have only the 
    stimulating electrode.

    Unlike the above class, manual analysis structs are cell specific, so
    any method for this class is assumed to be cell specific, too. 

    It's important to note that in manual analysis, the amplitude indices
    'finalized' by the human observer are written to this struct, but all, 
    including non-finalized amplitude data, are returned from these methods.
    Outside of this function, it will be important to index accordingly.

    Example usage:
        import elecresploader as el
        analysispath = '/Volumes/Analysis/2020-02-27-2/data001/'
        elecresp_name = 'elecResp_n7593_p507_r9.mat'
        er = el.ElecRespReader(analysispath,elecresp_name)
        amplitudes = er.get_amplitudes()

        ...

        See source code for more methods.
    '''

    def __init__(self,analysispath: str, elecresp_name: str) -> None:
        '''
        Constructor. Initializes object, and loads in file. Sets the file flag
        accordingly, returns if file doesn't exist.
        '''
        assert os.path.isdir(analysispath),"%s not a path"%analysispath
        self.elecresp_path = os.path.join(analysispath,
                                          elecresp_name)

        if not os.path.isfile(self.elecresp_path):
            print('%s not written in path'%elecresp_name)
            self.isfile = False

        self.isfile = True
        self.elecresp = loadmat(self.elecresp_path,
                                struct_as_record=False)

    def get_elecresp_dict(self) -> dict:
        '''
        Args:
            N/A

        Returns:
            Dictionary with several elecResp fields and their values.
        '''

        # Load in major fields from the struct.
        elecresp = self.elecresp
        elecresp = elecresp['elecResp']
        stiminfo = elecresp[0,0].stimInfo
        neuroninfo = elecresp[0,0].cells
        analysis = elecresp[0,0].analysis
        names = elecresp[0,0].names
         
        ''' Load in the subfields, containing manual analysis information. 
        This is organized by cell, as above.
        '''
        amplitudes = stiminfo[0,0].stimAmps * -1 # To be positive.
        stim_elec = stiminfo[0,0].electrodes
        rec_cells = neuroninfo[0,0].all
        rec_cells_templates = neuroninfo[0,0].allEIs

        # Get some path information.
        ei_path = names[0,0].rrs_ei_path[0]
        ei_path = os.path.dirname(ei_path)
        data_path = names[0,0].data_path[0]

        # The following is only pertinent for the cell ID passed.
        finalized = analysis[0,0].finalized
        spike_prob = analysis[0,0].successRates
        threshold = analysis[0,0].threshold
        spike_latencies = analysis[0,0].latencies
        est_artifact = analysis[0,0].estArtifact
        rec_elec = neuroninfo[0,0].recElec
        neuron_id = neuroninfo[0,0].main
        pulse_vectors = stiminfo[0,0].pulseVectors
        flags = analysis[0,0].details[0,0].analysisFlags

        # Save all of this as a dictionary, and return.
        elecresp_dict = dict()
        elecresp_dict['neuron_id'] = neuron_id
        elecresp_dict['amplitudes'] = amplitudes
        elecresp_dict['stim_elec'] = stim_elec
        elecresp_dict['rec_elec'] = rec_elec
        elecresp_dict['rec_cells'] = rec_cells
        elecresp_dict['rec_cells_templates'] = rec_cells_templates
        elecresp_dict['finalized'] = finalized
        elecresp_dict['spike_prob'] = spike_prob
        elecresp_dict['threshold'] = threshold
        elecresp_dict['spike_latencies'] = spike_latencies
        elecresp_dict['est_artifact'] = est_artifact
        elecresp_dict['pulse_vectors'] = pulse_vectors
        elecresp_dict['flags'] = flags
        elecresp_dict['ei_path'] = ei_path
        elecresp_dict['data_path'] = data_path
        return elecresp_dict

    def get_ei_path(self) -> str:
        '''
        Args:
            N/A

        Returns:
            String containing path to the white noise analysis used for 
            template matching.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['ei_path']

    def get_data_path(self) -> str:
        '''
        Args:
            N/A

        Returns:
            String containing path to the preprocessed electrical stimulation 
            data.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['data_path']

    def get_amplitudes(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Array of the (absolute) amplitudes, in µA.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['amplitudes'].flatten()

    def get_cell(self) -> int:
        '''
        Args:
            N/A

        Returns:
            Cell ID analyzed.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['neuron_id'][0][0]

    def get_stim_elec(self) -> np.ndarray or int:
        '''
        Args:
            N/A

        Returns:
            Stimulating electrode, or array of stimulating electrodes if
            applicable.
        '''
        elecresp_dict = self.get_elecresp_dict()
        stim_elecs = elecresp_dict['stim_elec'][0]

        # If there is only one, return it, otherwise return whole array. 
        if stim_elecs.shape[0] == 1: 
            return stim_elecs[0]

        return stim_elecs

    def get_rec_elec(self) -> int:
        '''
        Args:
            N/A

        Returns:
            Center recording electrode used during manual analysis.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['rec_elec'][0][0]

    def get_finalized(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Array of the finalized amplitudes (amplitude indexed). 1 if
            finalized, 0 otherwise.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['finalized'].flatten()

    def get_spike_prob(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Amplitude indexed array of response probabilities.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['spike_prob'].flatten()

    def get_spike_latencies(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Spike latencies at each trial, for each amplitude (nested array).

        Not totally clear on the units (samples?), but any trial with a
        non-zero value is when a spike occurred.

        Because of cleared values in the struct, and different trial numbers
        (sometimes), this array is nested.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['spike_latencies']

    def get_flags(self) -> np.ndarray:
        '''
        Args:
            N/A

        Returns:
            Flagged trials as marked as having some activity on them (putative
            spikes).

        Useful if one wanted to create their own artifact estimate using only
        failure traces, and unflagged trials. Another nested array.
        '''
        elecresp_dict = self.get_elecresp_dict()
        return elecresp_dict['flags']

    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        del self.elecresp
        return

    def close(self):
        del self.elecresp
        return
