import numpy as np

import visionloader.cython_extensions.visionfile_cext as vcext
import visionloader.visionload_cpp_extensions as vcppext
from bin2py import FakeArrayID, PyBinHeader
import electrode_map as elmap

import struct
import os
import io
from collections import namedtuple

from typing import Dict, Tuple, List, Optional, Any, Union, Set

import warnings

N_BYTES_BOOLEAN = 1
N_BYTES_16BIT = 2
N_BYTES_32BIT = 4
N_BYTES_64BIT = 8


def load_vision_data(analysis_path: str,
                     dataset_name: str,
                     include_params: bool = False,
                     include_ei: bool = False,
                     include_runtimemovie_params: bool = False,
                     include_sta: bool = False,
                     include_neurons: bool = False,
                     include_spikes: bool = False,
                     include_noise: bool = False,
                     include_model: bool = False,
                     parameters_extension: str = 'params',
                     globals_extension: str = 'globals',
                     ei_extension: str = 'ei',
                     sta_extension: str = 'sta',
                     neuron_extension: str = 'neurons',
                     spikes_extension: str = 'spikes',
                     noise_extension: str = 'noise',
                     model_extension: str = 'model') -> "VisionCellDataTable":
    '''
    Wrapper function for loading Vision data into Python
    Constructs and packs a VisionCellDataTable object with the specified data

    Args:
        analysis_path (str) : path to analysis folder (i.e. /Volumes/Analysis/01-01-2019/data000).
            Must be path to a folder
        dataset_name (str) : name of the datarun (i.e. data000)
        include_params (bool) : True if we should load data from parameters file, False otherwise
        include_runtimemovie_params (bool) : True if we should load data from the runtime movie parameters
            contained in the globals file, False otherwise
        include_sta (bool) : True if we should include the full STA movies for each cell, False otherwise
            Note that the STA fits can be accessed by loading the parameters file
        parameters_extension, globals_extension, ei_extension, sta_extension (str) : Optional,
            allows user to specify file extension for respective files


    Returns: VisionCellDataTable with relevant entries packed
    '''

    vcd = VisionCellDataTable()

    if include_runtimemovie_params:
        with GlobalsFileReader(analysis_path,
                               dataset_name,
                               globals_extension=globals_extension) as gfr:
            runtimemovie_params = gfr.get_run_time_movie_params()
            vcd.set_runtimemovie_params(runtimemovie_params)

    # if we are loading the params, we must load the params file before the EI file
    # Vision adds gibberish cell ids into some EI files for some reason
    #
    # The behavior if we load both params and EI files is that we load the params first,
    # and then only load EIs for cells that exist in the params file
    #
    # The behavior if we load only the EI file is to load everything in the EI file,
    # since in that case we have no idea what is legit and what isn't
    if include_params:
        with ParametersFileReader(analysis_path,
                                  dataset_name,
                                  parameters_extension=parameters_extension) as pfr:
            pfr.update_visioncelldata_obj(vcd)

    if include_ei:
        with EIReader(analysis_path,
                      dataset_name,
                      ei_extension=ei_extension,
                      globals_extension=globals_extension) as eir:
            eis_by_cell_id = eir.get_all_eis_by_cell_id()
            vcd.add_ei_from_loaded_ei_dict(eis_by_cell_id, restrict_to_existing_cells=include_params)
            vcd.set_electrode_map(eir.get_electrode_map())
            vcd.set_disconnected_electrodes(eir.get_disconnected_electrodes())

    if include_sta:
        with STAReader(analysis_path,
                       dataset_name,
                       sta_extension=sta_extension) as star:
            stas_by_cell_id = star.chunked_load_all_stas()
            vcd.add_sta_from_loaded_sta_dict(stas_by_cell_id)

    if include_neurons:
        with NeuronsReader(analysis_path,
                           dataset_name,
                           neuron_extension=neuron_extension) as nr:
            spike_samples_by_neuron_id = nr.get_spike_sample_nums_for_all_real_neurons()
            ttl_times = nr.get_TTL_times()
            n_samples = nr.n_samples
            vcd.add_spike_times_from_loaded_spike_times_dict(spike_samples_by_neuron_id,
                                                             ttl_times,
                                                             n_samples)

    '''
    To avoid confusion, this loads ALL spike times on a given electrode,
    whereas the other spikes method only adds spike times for a given cell.
    '''
    if include_spikes:
        with SpikesReader(analysis_path,
                          dataset_name,
                          spikes_extension=spikes_extension) as sr:
            spike_times_by_electrode = sr.get_spiketimes_by_electrode()
            vcd.add_spike_times_by_electrode(spike_times_by_electrode)

    if include_noise:
        with NoiseReader(analysis_path,
                         dataset_name,
                         noise_extension=noise_extension) as rmsr:
            channel_noise = rmsr.get_channel_noise()
            vcd.add_channel_noise(channel_noise)

    if include_model:
        with ModelReader(analysis_path,
                         dataset_name,
                         model_extension=model_extension) as mr:
            raw_neurons_map = mr.get_raw_neurons_by_electrode()
            vcd.add_raw_neurons_map(raw_neurons_map)

    return vcd


'''
Used to store and return STA fits
All attributes should be float
'''
STAFit = namedtuple('STAFit', ['center_x', 'center_y', 'std_x', 'std_y', 'rot'])


class VisionFieldNames:
    '''
    Defines attribute column names for VisionCellDataTable

    Has direct  correspondence with the column names
        that are loaded from the parameters file
        so do not change these values unless you make
        the corresponding changes in the Vision source code

    EI_FIELDNAME, SPIKE_TIMES_FIELDNAME , ELEC_SPIKE_TIMES_FIELDNAME,
    STA_FIELD_NAME, and MODEL_FIELD_NAME are defined only within this module, 
    so these can be changed.

    
    '''
    CLASSID_FIELDNAME = 'classID'
    CELLID_FIELDNAME = 'ID'

    EI_FIELDNAME = 'EI'

    STA_FIELDNAME = 'STAraw'

    SPIKE_TIMES_FIELDNAME = 'SpikeTimes'
    ELEC_SPIKE_TIMES_FIELDNAME = 'ElecSpikeTimes'
    CHANNEL_NOISE_FIELDNAME = 'ChannelNoise'
    MODEL_FIELD_NAME = 'Model'

    STA_FITX_STD_FIELDNAME = 'SigmaX'
    STA_FITY_STD_FIELDNAME = 'SigmaY'
    STA_XCENTER_FIELDNAME = 'x0'
    STA_YCENTER_FIELDNAME = 'y0'
    STA_ROT_FIELDNAME = 'Theta'

    CONTAMINATION_FIELDNAME = 'contamination'

    ACF_FIELDNAME = 'acfMean'
    ACF_NUMPAIRS_FIELDNAME = 'Auto'


class VisionCellDataTable:
    '''
    Class that stores the data loaded from a Vision analysis

    Use this to access data from a Vision analysis in Python
    '''

    def __init__(self) -> None:

        '''
        Constructor, initializes an empty object
        '''

        self.main_datatable = {}  # type: Dict[int, Dict[str, Any]]
        # key is cell id
        # value is another dict with attributes
        # example structure: { 1234 : { 'cellID' : 1234, 'classID' : 'ON parasol', ... }}

        self.all_field_names = set()  # type: Set[str]
        # items in here have type str,
        # mostly corresponding to the column names in the parameters file
        # with a few added ones for EI and full STA

        self.electrode_map = None  # np array of shape (n_electrodes, 2),
        # containing the coordinates of the electrodes
        # doesn't include TTL channel

        self.disconnected_electrodes = set()  # type: Set[int]

        self.ttl_times = None  # type: Union[np.ndarray, None] # either a 1D np.ndarray with ttl times
        # if we loaded the .neurons file, or None if we didn't
        self.n_samples = None  # type: Union[int, None] # either a positive integer telling us how many
        # samples there were if we loaded the.neurons file, or None if we didn't

        self.runtimemovie_params = None  # type: Optional[RunTimeMovieParamsReader]

        self.electrode_spike_times = None  # type dict[int,dict] if loaded .spikes.

    def get_n_samples(self) -> int:
        assert self.n_samples is not None, "Load .neurons for n_samples"
        return self.n_samples

    def get_ttl_times(self) -> np.ndarray:

        assert self.ttl_times is not None, "No TTL times, load .neurons for TTL times"
        return self.ttl_times

    def get_all_present_cell_types(self) -> List[str]:

        '''
        Gets all of the distinct cell types in the dataset

        :return: List of str, corresponding to the different cell type names
        '''

        cell_types_set = set()  # type: Set[str]
        for cell_id, cell_data in self.main_datatable.items():
            cell_type = cell_data[VisionFieldNames.CLASSID_FIELDNAME]
            if cell_type not in cell_types_set:
                cell_types_set.add(cell_type)
        return list(cell_types_set)

    def get_all_cells_of_type(self,
                              class_name: str) -> List[int]:

        '''
        Gets all of the cells with cell type class_name

        args:
            class_name (str) : class of cells that we want to get

        Returns:
            list of int, corresponding to the cell ids of every cell
                with class class_name
        '''

        cell_ids = []
        for cell_id, cell_data in self.main_datatable.items():
            if cell_data[VisionFieldNames.CLASSID_FIELDNAME] == class_name:
                cell_ids.append(cell_id)
        return cell_ids

    def get_all_cells_similar_to_type(self,
                                      class_name: str) -> List[int]:
        '''
        Gets all of the cells with cell types closely matching in name the cell type class_name
        args:
            class_name (str) : class of cells that we want to get
        Returns:
            list of int, corresponding to the cell ids of every cell
                with class class_name
        '''
        cell_ids = []
        class_name = class_name.lower()
        if class_name[-1] == 's':
            class_name = class_name[:-1]
        for cell_id, cell_data in self.main_datatable.items():
            if class_name in (cell_data[VisionFieldNames.CLASSID_FIELDNAME]).lower():
                cell_ids.append(cell_id)
        return cell_ids

    def get_cell_ids(self) -> List[int]:
        '''
        Gets the cell ids of every cell that has information stored in the object

        Returns:
            list of int, corresponding to all of the cell ids
        '''
        return list(self.main_datatable.keys())

    def get_all_field_names(self) -> List[str]:
        '''
        Gets all of the field (data table column) names stored in the object

        Useful for figuring out what data has been loaded into the object

        Returns:
            list of str, corresponding to the different field names
        '''
        return list(self.all_field_names)

    def set_runtimemovie_params(self,
                                rtmp: 'RunTimeMovieParamsReader') -> None:
        '''
        Stores the RunTimeMovieParamsReader object

        Args:
            rtmp (RunTimeMovieParamsReader)
        '''
        self.runtimemovie_params = rtmp  # type: RunTimeMovieParamsReader

    def get_runtimemovie_params(self) -> Optional['RunTimeMovieParamsReader']:
        '''
        Gets the RunTimeMovieParamsReader object

        Returns:
            RunTimeMovieParamsReader object
        '''
        return self.runtimemovie_params

    def set_electrode_map(self,
                          electrode_map: np.ndarray) -> None:
        '''
        Sets the electrode map

        Args:
            electrode_map (np array) : np array of shape (n_electrodes, 2)
                corresponding to the coordinates of each electrode. Does
                not include the TTL channel, if you include the TTL channel
                things will break
        '''
        self.electrode_map = electrode_map

    def get_electrode_map(self) -> np.ndarray:
        '''
        Gets the electrode map

        Returns:
            np array corresponding to the electrode map
        '''
        return self.electrode_map

    def set_disconnected_electrodes(self,
                                    disconnected_electrodes: Set[int]) -> None:
        self.disconnected_electrodes = disconnected_electrodes

    def get_disconnected_electrodes(self) -> Set[int]:
        return self.disconnected_electrodes

    def update_data_for_cell_id_and_field_name(self,
                                               cell_id: int,
                                               field_name: str,
                                               data: Any) -> None:

        '''
        Updates the entry for a particular field name (attribute) for a particular cell

        If no data for the cell exists in the data structure, add an entry
            for that cell
        If the field name does not exist, add that
            field name

        Overwrites existing data if there is any

        Args:
            cell_id (int) : id of the cell
            field_name (str) : field name (attribute, or data table column name)
            data (any type) : data that we want to store for the particular
                cell_id and field_name
        '''

        if field_name not in self.all_field_names:
            self.all_field_names.add(field_name)

        if cell_id not in self.main_datatable:
            self.main_datatable[cell_id] = {}

        self.main_datatable[cell_id][field_name] = data

    def add_ei_from_loaded_ei_dict(self,
                                   ei_by_cell_id: Dict[int, 'EIContainer'],
                                   restrict_to_existing_cells: bool = True) -> None:
        '''
        Stores EIs in the data structure. Each EI must be
            associated with a cell id

        Args:
            ei_by_cell_id (dict) : dict with format
                { cell_id (int) : EIContainer namedtuple }
        '''

        self.all_field_names.add(VisionFieldNames.EI_FIELDNAME)

        for cell_id, ei in ei_by_cell_id.items():
            if restrict_to_existing_cells and cell_id in self.main_datatable:
                self.main_datatable[cell_id][VisionFieldNames.EI_FIELDNAME] = ei
            elif not restrict_to_existing_cells and cell_id not in self.main_datatable:
                self.main_datatable[cell_id] = {}
                self.main_datatable[cell_id][VisionFieldNames.EI_FIELDNAME] = ei

    def add_spike_times_from_loaded_spike_times_dict(self,
                                                     spike_times_by_cell_id: Dict[int, np.ndarray],
                                                     ttl_times: np.ndarray,
                                                     n_samples_dataset: int) -> None:

        self.all_field_names.add(VisionFieldNames.SPIKE_TIMES_FIELDNAME)
        for cell_id, spike_time_array in spike_times_by_cell_id.items():
            if cell_id not in self.main_datatable:
                self.main_datatable[cell_id] = {}

            self.main_datatable[cell_id][VisionFieldNames.SPIKE_TIMES_FIELDNAME] = spike_time_array

        self.ttl_times = ttl_times
        self.n_samples = n_samples_dataset

    def add_spike_times_by_electrode(self,
                                     spike_times_by_electrode:
                                     dict) -> None:
        self.all_field_names.add(VisionFieldNames.ELEC_SPIKE_TIMES_FIELDNAME)
        self.electrode_spike_times = spike_times_by_electrode

    def add_raw_neurons_map(self,
                            raw_neurons_map: dict) -> None:
        self.all_field_names.add(VisionFieldNames.MODEL_FIELD_NAME)
        self.raw_neurons_map = raw_neurons_map

    def add_channel_noise(self,
                          channel_noise: np.ndarray) -> None:
        self.all_field_names.add(VisionFieldNames.CHANNEL_NOISE_FIELDNAME)
        self.channel_noise = channel_noise

    def add_sta_from_loaded_sta_dict(self,
                                     sta_by_cell_id: Dict[int, 'STAContainer']) -> None:
        '''
        Stores full STAs in the data structure. Each STA must be
            associated with a cell id

        Args:
            sta_by_cell_id (dict) : dict with format
                { cell_id (int) : sta (np array) }
        '''

        self.all_field_names.add(VisionFieldNames.STA_FIELDNAME)
        for cell_id, sta_tup in sta_by_cell_id.items():
            if cell_id not in self.main_datatable:
                self.main_datatable[cell_id] = {}
            self.main_datatable[cell_id][VisionFieldNames.STA_FIELDNAME] = sta_tup

    def get_cell_type_for_cell(self,
                               cell_id: int) -> str:
        '''
        Gets the cell type for a given cell id

        Args:
            cell_id (int) : the cell id
        Returns:
            str : cell type name
        '''

        assert cell_id in self.main_datatable, "Cell id {0} not found in dataset".format(cell_id)
        return self.main_datatable[cell_id][VisionFieldNames.CLASSID_FIELDNAME]

    def get_all_data_for_cell(self,
                              cell_id: int) -> Dict:
        '''
        Gets all of the stored attributes for a given cell id

        Note: it would be a good idea to avoid modifying the the contents
            of the return value, because that would modify the contents
            of the data structure, and possibly lead to bad things

        Args:
            cell_id (int) : the cell id

        Returns:
            dict of format {attribute (str) : value (any type)} corresponding
                to all of the data stored for cell_id
        '''
        assert cell_id in self.main_datatable, "Cell id {0} not found in dataset".format(cell_id)
        return self.main_datatable[cell_id]

    def get_data_for_cell(self,
                          cell_id: int,
                          field_name: str) -> Any:
        '''
        Gets the data stored for a particular field name for a particular cell id

        Args:
            cell_id (int) : cell id
            field_name (str) : field name / attribute / column name  to get

        Returns:
            any type, corresponding to the data stored for cell cell_id at field name field_name
        '''

        assert cell_id in self.main_datatable, "Cell id {0} not found in dataset".format(cell_id)
        assert field_name in self.main_datatable[cell_id], "{0} not found for cell {1}".format(field_name, cell_id)

        return self.main_datatable[cell_id][field_name]

    def get_spike_times_for_cell(self,
                                 cell_id: int) -> np.ndarray:
        '''
        Gets the spike times (sample number) for a cell

        Args:
            cell_id (int) : cell id

        Returns:
            np.array of int, with shape (n_spikes, ) where n_spikes is the number of times the
                cell spiked. The integers correspond to the sample number in the raw data that
                the cell spiked at.
        '''
        return self.get_data_for_cell(cell_id, VisionFieldNames.SPIKE_TIMES_FIELDNAME)

    def get_spiketimes_all_electrodes(self) -> dict:
        '''
        Gets the spike times for ALL electrodes.

        Returns:
            Dict (int mapping to spike times and counts).
        '''
        return self.electrode_spike_times

    def get_ei_for_cell(self,
                        cell_id: int) -> "EIContainer":
        '''
        Gets the EI for a cell

        Args:
            cell_id (int) : cell id

        Returns:
            EIContainer namedtuple, containing the EI and associated parameters. Avoid modifying
                the contents of this array, as this will change the EI stored by the data structure
        '''

        return self.get_data_for_cell(cell_id, VisionFieldNames.EI_FIELDNAME)

    def get_contamination_for_cell(self,
                                   cell_id: int) -> float:
        '''
        Gets the contamination for a cell

        Args:
            cell_id (int) : cell_id

        Returns:
            float, the contamination value for the cell
        '''
        return self.get_data_for_cell(cell_id, VisionFieldNames.CONTAMINATION_FIELDNAME)

    def get_acf_numpairs_for_cell(self, cell_id: int) -> np.ndarray:
        '''
        Gets the ACF number of pairs curve for a cell
            Note that this the correct one, the below method does not return an array


        Args:
            cell_id (int) : cell id

        Returns:
            np.array of float containing ACF function. Note that the y-axis scaling of the ACF
                function may be different than that of Vision
        '''

        return self.get_data_for_cell(cell_id, VisionFieldNames.ACF_NUMPAIRS_FIELDNAME)

    def get_acf_for_cell(self,
                         cell_id: int) -> np.ndarray:
        '''
        BUGGED!!! FIXME
        Gets the acf for a cell

        Args:
            cell_id (int) : cell id

        Returns:
            np.array of float containing ACF function. Note that the y-axis scaling of the ACF
                function may be different than that of Vision
        '''
        warnings.warn('get_acf_for_cell deprecated, use get_acf_numpairs_for_cell', DeprecationWarning)
        return self.get_data_for_cell(cell_id, VisionFieldNames.ACF_FIELDNAME)

    def get_class_for_cell(self,
                           cell_id: int) -> str:
        '''
        Gets the class for a cell

        Args:
            cell_id (int) : cell id
        Returns:
            str, corresponding to class of the cell
        '''
        return self.get_data_for_cell(cell_id, VisionFieldNames.CLASSID_FIELDNAME)

    def get_sta_for_cell(self,
                         cell_id: int) -> np.ndarray:
        '''
        Gets the STA for a cell

        Args:
            cell_id (int) : cell id

        Returns:
            np.array of float, with shape (width, height, n_frames). Avoid modifying this array
                as this will change the STA stored by the data structure
        '''
        return self.get_data_for_cell(cell_id, VisionFieldNames.STA_FIELDNAME)

    def get_stafit_for_cell(self,
                            cell_id: int) -> 'STAFit':

        '''
        Gets the STA Gaussian fit (center, standard deviations, rotation) for a cell

        Note the bizarre hack at the end; that is match the coordinates given by
            MATLAB and Vision. Since it only shifts the data, it could be done
            away with in the future

        Args:
            cell_id (int) : cell id

        Returns:
            STAFit : namedtuple containing floats for each of the fit parameters
        '''

        x_center = self.get_data_for_cell(cell_id,
                                          VisionFieldNames.STA_XCENTER_FIELDNAME)
        y_center = self.get_data_for_cell(cell_id,
                                          VisionFieldNames.STA_YCENTER_FIELDNAME)

        x_std = self.get_data_for_cell(cell_id,
                                       VisionFieldNames.STA_FITX_STD_FIELDNAME)
        y_std = self.get_data_for_cell(cell_id,
                                       VisionFieldNames.STA_FITY_STD_FIELDNAME)

        rot = self.get_data_for_cell(cell_id,
                                     VisionFieldNames.STA_ROT_FIELDNAME)

        if self.runtimemovie_params is None:
            return STAFit(x_center, y_center, x_std, y_std, rot)

        else:  # match output of MATLAB load_sta
            return STAFit(x_center + 0.5,
                          self.runtimemovie_params.height - y_center + 0.5,
                          x_std,
                          y_std,
                          rot)

    def update_cell_type_classifications_from_text_file(self,
                                                        path_to_text_file: str) -> None:

        '''
        Updates the data structure to reflect the cell type classification saved
            in a text file


        Args:
            path_to_text_file (str) : path of the text file saved by Vision
        '''

        assert os.path.isfile(path_to_text_file)
        with open(path_to_text_file, 'r') as class_text_file:

            for line in class_text_file:
                values = line.split()

                cell_id = int(values[0])

                cell_type = ' '.join(values[1:])[:-1]
                if cell_type.startswith('All/'):
                    cell_type = cell_type.replace('All/', '', 1)
                if cell_type == 'All':
                    cell_type = 'Unclassified'
                cell_type = cell_type.replace('/', ' ')
                cell_type = cell_type.replace('-', ' ')

                self.update_data_for_cell_id_and_field_name(cell_id,
                                                            VisionFieldNames.CLASSID_FIELDNAME,
                                                            cell_type)


class ChunkFileReader:
    '''
    Base class used to read files that are based on ChunkFile in the
        Vision source code.

    Basic idea: the file has different chunks. Each chunk is identified by
        a 96 bit tag, which contains the chunk ID and the size of the data.
        The data for a chunk is stored immediately after the tag.

        The chunk ID tells you what data is stored in the tag, and the size
        tells you how many bytes are in the data section.

    Currently this only used to read the globals file.

    Don't directly use this class unless you know what you're doing
    '''

    def __init__(self, cf_path: str) -> None:

        self.fp = open(cf_path, 'rb')

    '''
    Goes through the file and searches for a chunk whose tag matches
        the specified tag. If there is a match, return the contents
        corresponding to the tag as a bytearray

    Args:
        tag (int) : tag to look for

    Returns:
        int, num bytes contained in the content
        bytearray, corresponding to the content of the part of the
            file that matches the tag. Does not include the tag.
    '''

    def get_chunk_bytearray(self,
                            tag: int) -> Tuple[int, Union[bytearray, bytes]]:

        # first go to the beginning of the file
        self.fp.seek(N_BYTES_64BIT, 0)  # need to skip file id

        tag_size_asbytearray = self.fp.read(N_BYTES_32BIT * 3)
        if len(tag_size_asbytearray) != N_BYTES_32BIT * 3:
            raise IOError()
        curr_tag, _, curr_size = struct.unpack('>III', tag_size_asbytearray)  # type: int, int, int

        while curr_tag != tag:
            self.fp.seek(curr_size, 1)

            tag_size_asbytearray = self.fp.read(N_BYTES_32BIT * 3)

            if len(tag_size_asbytearray) != N_BYTES_32BIT * 3:
                raise IOError()

            curr_tag, _, curr_size = struct.unpack('>III', tag_size_asbytearray)

        content = self.fp.read(curr_size)
        return curr_size, content

    def read_chunk_pair_float_array_dtype(self,
                                          tag: int) -> np.ndarray:
        '''
        Finds chunk with
        '''

        payload_size, content_as_bytearray = self.get_chunk_bytearray(tag)

        assert payload_size % (N_BYTES_32BIT * 2) == 0, "Size is not divisble by 8, \
                content cannot be pairs of 32-bit float"

        num_pairs_to_read = payload_size >> 3
        output = np.zeros((num_pairs_to_read, 2), dtype=np.float32)
        vcext.unpack_alternating_32bit_float_from_bytearray(content_as_bytearray, num_pairs_to_read, output)

        return output

    def has_tag(self,
                tag: int) -> bool:

        self.fp.seek(N_BYTES_64BIT, 0)  # need to skip file id

        # then loop through the tags
        curr_tag, curr_size = None, None

        tag_size_asbytearray = self.fp.read(N_BYTES_32BIT * 3)
        if len(tag_size_asbytearray) != N_BYTES_32BIT * 3:
            return False
        curr_tag, _, curr_size = struct.unpack('>III', tag_size_asbytearray)

        while curr_tag != tag:
            self.fp.seek(curr_size, 1)

            tag_size_asbytearray = self.fp.read(N_BYTES_32BIT * 3)

            if len(tag_size_asbytearray) != N_BYTES_32BIT * 3:
                return False

            curr_tag, _, curr_size = struct.unpack('>III', tag_size_asbytearray)

        return True

    def read_chunk_int_dtype(self, tag: int) -> int:

        payload_size, content_as_bytearray = self.get_chunk_bytearray(tag)
        assert payload_size == 4, "Content is not a single 32-bit integer"
        return struct.unpack('>i', content_as_bytearray)[0]

    def read_chunk_float64_dtype(self, tag: int) -> float:

        payload_size, content_as_bytearray = self.get_chunk_bytearray(tag)
        assert payload_size == N_BYTES_64BIT, "Content is not a single 64-bit float"
        return struct.unpack('>d', content_as_bytearray)[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fp.close()

    def close(self):
        self.fp.close()


class DefaultVisionParams:
    # in ms
    ACFT1 = 0.5  # type: float

    # in ms
    ACFT2 = 1.0  # type: float

    SAMPLES_PER_MILLISECOND = 20  # type: int

    DEFAULT_MONITOR_FREQUENCY = 120.0  # type: float
    DEFAULT_FRAMES_PER_TTL = 100  # type: int


class ImageCalibrationParamsReader:

    def __init__(self,
                 microns_per_pixel_x: float,
                 microns_per_pixel_y: float,
                 center_x: float,
                 center_y: float,
                 flip_x: bool,
                 flip_y: bool,
                 angle: float,
                 array_id: int,
                 array_part: int,
                 array_n_parts: int) -> None:
        self.microns_per_pixel_x = microns_per_pixel_x
        self.microns_per_pixel_y = microns_per_pixel_y

        self.center_x = center_x
        self.center_y = center_y

        self.flip_x = flip_x
        self.flip_y = flip_y
        self.angle = angle

        self.array_id = array_id
        self.array_part = array_part
        self.array_n_parts = array_n_parts

    @classmethod
    def construct_from_bytearray(cls,
                                 byte_array: Union[bytes, bytearray]) -> 'ImageCalibrationParamsReader':
        bstream = io.BytesIO(byte_array)

        microns_per_pixel_x, microns_per_pixel_y = struct.unpack('>dd', bstream.read1(N_BYTES_64BIT * 2))

        center_x, center_y = struct.unpack('>dd', bstream.read1(N_BYTES_64BIT * 2))

        flip_x, flip_y = struct.unpack('>??', bstream.read1(N_BYTES_BOOLEAN * 2))

        angle = struct.unpack('>d', bstream.read1(N_BYTES_64BIT))[0]

        array_id, array_part, array_n_parts = struct.unpack('>iii',
                                                            bstream.read1(N_BYTES_32BIT * 3))

        return ImageCalibrationParamsReader(microns_per_pixel_x,
                                            microns_per_pixel_y,
                                            center_x,
                                            center_y,
                                            flip_x,
                                            flip_y,
                                            angle,
                                            array_id,
                                            array_part,
                                            array_n_parts)

    def to_bytearray(self) -> bytes:
        return struct.pack('>dddd??diii',
                           self.microns_per_pixel_x,
                           self.microns_per_pixel_y,
                           self.center_x,
                           self.center_y,
                           self.flip_x,
                           self.flip_y,
                           self.angle,
                           self.array_id,
                           self.array_part,
                           self.array_n_parts)


class RunTimeMovieParamsReader:
    '''
    Class that loads and stores the RunTimeMovieParams from data contained in the
        globals file, if it exists

    This contains information about the white noise stimulus
    '''

    def __init__(self,
                 pixelsPerStixelX: int,
                 pixelsPerStixelY: int,
                 width: int,
                 height: int,
                 micronsPerStixelX: float,
                 micronsPerStixelY: float,
                 xOffset: float,
                 yOffset: float,
                 interval: int,
                 monitorFrequency: float,
                 framesPerTTL: int,
                 refreshPeriod: float,
                 nFramesRequired: int,
                 droppedFrames: List[int]) -> None:

        self.pixelsPerStixelX = pixelsPerStixelX
        self.pixelsPerStixelY = pixelsPerStixelY
        self.width = width
        self.height = height
        self.micronsPerStixelX = micronsPerStixelX
        self.micronsPerStixelY = micronsPerStixelY
        self.xOffset = xOffset
        self.yOffset = yOffset
        self.interval = interval
        self.monitorFrequency = monitorFrequency
        self.framesPerTTL = framesPerTTL
        self.refreshPeriod = refreshPeriod
        self.nFramesRequired = nFramesRequired
        self.droppedFrames = droppedFrames

    @classmethod
    def construct_from_bytearray(cls,
                                 rtmp_bytearray_chunk: Union[bytes, bytearray],
                                 mf_bytearray_chunk: Optional[Union[bytes, bytearray]] = None) \
            -> 'RunTimeMovieParamsReader':

        bstream = io.BytesIO(rtmp_bytearray_chunk)

        pixelsPerStixelX, pixelsPerStixelY = struct.unpack('>ii', bstream.read1(N_BYTES_32BIT * 2))
        width, height = struct.unpack('>dd', bstream.read1(N_BYTES_64BIT * 2))
        micronsPerStixelX, micronsPerStixelY = struct.unpack('>dd', bstream.read1(N_BYTES_64BIT * 2))
        xOffset, yOffset = struct.unpack('>dd', bstream.read1(N_BYTES_64BIT * 2))
        interval = struct.unpack('>i', bstream.read1(N_BYTES_32BIT))[0]
        refreshPeriod = struct.unpack('>d', bstream.read1(N_BYTES_64BIT))[0]
        nFramesRequired = struct.unpack('>i', bstream.read1(N_BYTES_32BIT))[0]

        num_dropped_frames = struct.unpack('>i', bstream.read1(N_BYTES_32BIT))[0]
        dropped_frames = []
        for i in range(num_dropped_frames):
            dropped_frames.append(struct.unpack('>i', bstream.read1(N_BYTES_32BIT))[0])

        monitorFrequency = DefaultVisionParams.DEFAULT_MONITOR_FREQUENCY
        framesPerTTL = DefaultVisionParams.DEFAULT_FRAMES_PER_TTL

        if mf_bytearray_chunk is not None:
            monitorFrequency = struct.unpack('>d', mf_bytearray_chunk[0:N_BYTES_64BIT])[0]
            framesPerTTL = struct.unpack('>i', mf_bytearray_chunk[N_BYTES_64BIT:N_BYTES_64BIT + N_BYTES_32BIT])[0]

        return RunTimeMovieParamsReader(
            pixelsPerStixelX,
            pixelsPerStixelY,
            width,
            height,
            micronsPerStixelX,
            micronsPerStixelY,
            xOffset,
            yOffset,
            interval,
            monitorFrequency,
            framesPerTTL,
            refreshPeriod,
            nFramesRequired,
            dropped_frames)

    def generate_rtmp_in_binary(self) -> Tuple[bytes, bytes]:

        rtmp_front = struct.pack('>iiddddddidi',
                                 self.pixelsPerStixelX,
                                 self.pixelsPerStixelY,
                                 self.width,
                                 self.height,
                                 self.micronsPerStixelX,
                                 self.micronsPerStixelY,
                                 self.xOffset,
                                 self.yOffset,
                                 self.interval,
                                 self.refreshPeriod,
                                 self.nFramesRequired)

        dropped_frame_bytes_list = [struct.pack('>i', len(self.droppedFrames))]
        for dropped_frame in self.droppedFrames:
            dropped_frame_bytes_list.append(struct.pack('>i', dropped_frame))

        dropped_frame_section = b''.join(dropped_frame_bytes_list)

        mf_bytearray_section = struct.pack('>di', self.monitorFrequency, self.framesPerTTL)

        return b''.join([rtmp_front, dropped_frame_section]), mf_bytearray_section


class GlobalsFileReader(ChunkFileReader):
    '''
    Class for reading the globals file

    Also contains code to figure out what the underlying electrode map is, based
        on the array_id in the globals file.
    If the array_id corresponds to Hierlemann data (arbitrary electrode geometry
        specified by the user, i.e. array_id=9999), then electrode map can be
        directly read from the globals file according to a Vision modification
        made by Georges.
    '''

    # TAGS
    DEFAULT = 0  # type: int
    UPDATER = 1  # type: int

    ICP_TAG = 0  # type: int
    RTMP_TAG = 1  # type: int
    CREATED_BY_TAG = 1  # type: int
    VERSION_TAG = 3  # type: int
    MONITOR_FREQ_TAG = 4  # type: int
    RDH512_TAG = 5  # type: int
    ARRAYID_TAG = 6  # type: int
    ELECTRODEMAP_TAG = 7  # type: int
    ELECTRODEPITCH_TAG = 8  # type: int

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 globals_extension: str = 'globals') -> None:

        assert os.path.isdir(analysis_folder_path)
        globals_file_path = os.path.join(analysis_folder_path, "{0}.{1}".format(dataset_name, globals_extension))
        assert os.path.isfile(globals_file_path)

        super().__init__(globals_file_path)

    def get_run_time_movie_params(self) -> RunTimeMovieParamsReader:

        '''
        Loads the RunTimeMovieParams from the globals file
        '''

        has_rtmp_tag = self.has_tag(GlobalsFileReader.RTMP_TAG)
        has_mf_tag = self.has_tag(GlobalsFileReader.MONITOR_FREQ_TAG)

        if not has_rtmp_tag:
            assert False, "Globals file does not have RTMP tag, cannot load runtime movie parameters"

        if has_rtmp_tag and has_mf_tag:
            _, rtmp_chunk = self.get_chunk_bytearray(GlobalsFileReader.RTMP_TAG)
            _, mf_chunk = self.get_chunk_bytearray(GlobalsFileReader.MONITOR_FREQ_TAG)

            return RunTimeMovieParamsReader.construct_from_bytearray(rtmp_chunk, mf_chunk)
        else:
            _, rtmp_chunk = self.get_chunk_bytearray(GlobalsFileReader.RTMP_TAG)
            return RunTimeMovieParamsReader.construct_from_bytearray(rtmp_chunk)

    def get_image_calibration_params(self) -> ImageCalibrationParamsReader:

        '''
        Loads the ImageCalibrationParams from the globals file, if it exists
        '''

        has_icp_tag = self.has_tag(GlobalsFileReader.ICP_TAG)
        if not has_icp_tag:
            assert False, "Globals file does not have ICP tag, cannot load image calibration parameters"

        _, icp_chunk = self.get_chunk_bytearray(GlobalsFileReader.ICP_TAG)

        return ImageCalibrationParamsReader.construct_from_bytearray(icp_chunk)

    def get_rdh512_header (self) -> Optional[PyBinHeader]:

        has_rdh512_tag = self.has_tag(GlobalsFileReader.RDH512_TAG)
        if has_rdh512_tag:
            _, header_as_bytearray = self.get_chunk_bytearray(GlobalsFileReader.RDH512_TAG)
            header_pybinheader = PyBinHeader.construct_from_bytearray(header_as_bytearray)
            return header_pybinheader
        return None

    def get_electrode_map(self) -> Tuple[np.ndarray, Set[int]]:

        '''
        Figures out what the electrode map is based on information in the globals file

        Requires that the array_id stored in the globals file be correct, otherwise
            terrible things will happen

        Determines electrode map in the following order:
        1. Checks the array_id tag, if it's valid then use the electrode map for that
            array_id
        2. Checks the rdh512 tag (which has the same format as the .bin file header) and
            tries to extract array id from that
        3. Checks the icp tag (corresponding to image calibration data) and tries to
            extract array id from that

        If the array id corresponds to a standard Litke array, then returns an electrode map
            corresponding to that array id

        If the array id is 9999, corresponding to arbitrary electrode configuration, then
            attempt to load the electrode map from the globals file itself

        Returns:
            np array of float, shape (n_electrodes, 2), where n_electrodes is the number of
                electrodes not including the TTL channel
        '''

        # TTL channel not included in the electrode map
        # based heavily on getElectrodeMap() from modified Vision source code
        has_array_id_tag = self.has_tag(GlobalsFileReader.ARRAYID_TAG)
        has_rdh512_tag = self.has_tag(GlobalsFileReader.RDH512_TAG)
        has_icp_tag = self.has_tag(GlobalsFileReader.ICP_TAG)

        if has_array_id_tag or has_rdh512_tag or has_icp_tag:

            array_id = -1

            if has_array_id_tag:

                array_id = self.read_chunk_int_dtype(GlobalsFileReader.ARRAYID_TAG) & 0xFFFF

            elif has_rdh512_tag:

                _, header_as_bytearray = self.get_chunk_bytearray(GlobalsFileReader.RDH512_TAG)
                header_pybinheader = PyBinHeader.construct_from_bytearray(header_as_bytearray)

                array_id = header_pybinheader.array_id

            elif has_icp_tag:
                _, icp_as_bytearray = self.get_chunk_bytearray(GlobalsFileReader.ICP_TAG)

                icp_obj = ImageCalibrationParamsReader.construct_from_bytearray(icp_as_bytearray)
                array_id = icp_obj.array_id

            if array_id == FakeArrayID.BOARD_ID_RECONFIGURABLE:
                if self.has_tag(GlobalsFileReader.ELECTRODEMAP_TAG):
                    coords_with_ttl = self.read_chunk_pair_float_array_dtype(GlobalsFileReader.ELECTRODEMAP_TAG)

                    '''
                    if self.has_tag(GlobalsFileReader.ELECTRODEPITCH_TAG):
                        electrode_pitch = self.read_chunk_float64_dtype(GlobalsFileReader.ELECTRODEPITCH_TAG)
                        coords_with_ttl = coords_with_ttl * electrode_pitch
                    '''

                    return coords_with_ttl[1:, :], elmap.get_disconnected_electrode_set_by_array_id(
                        FakeArrayID.BOARD_ID_RECONFIGURABLE)

                else:
                    assert False, "Reconfigurable electrode map not specified in globals file"
            else:
                return elmap.get_litke_array_coordinates_by_array_id(
                    array_id), elmap.get_disconnected_electrode_set_by_array_id(array_id)

        else:
            assert False, "No array id in globals file, no RDH512 header, and no image calibration parameters"


'''
container to store the arrays corresponding to EI
also stores number of left samples, number of right samples

ei : np.array of float with shape (n_electrodes, n_samples) where n_electrodes is the number of
        electrodes, not including the TTL channel
    contains the EI
ei_error : np.array of float with shape (n_electrodes, n_samples) where n_electrodes is the numer of
        electrodes, not including the TTL channel
    contains the EI error as calculated by Vision

nl_points : int, number of left points taken in the EI calculation
nr_points : int, number of right points taken in the EI calculation
n_samples_total : int, equal to nl_points + nr_points + 1
'''
EIContainer = namedtuple('EIContainer',
                         ['ei', 'ei_error', 'nl_points', 'nr_points', 'n_samples_total', 'n_spikes'])


class EIReader:
    '''
    Class for reading the EI file from Vision

    Example usage:

    with EIReader('/Volumes/Analysis/2019-01-01-1/data000', 'data000') as eir:
        eis_by_cell_id = eir.get_all_eis_by_cell_id()


    Alternative usage:

    eir = EIReader('/Volumes/Analysis/2019-01-01-1/data000', 'data000')
    eis_by_cell_id = eir.get_all_eis_by_cell_id()
    eir.close()
    '''

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 ei_extension: str = 'ei',
                 globals_extension: str = 'globals') -> None:

        '''
        Reads the header of the EI file to get information about the EIs

        Also reads the seek table to figure out where in the file the data
            for each EI is stored.

        Requires that the electrode map can be figured out from the globals file

        Note: Does not actually read the EIs. That is done elsewhere.
        '''

        assert os.path.isdir(analysis_folder_path), \
            "Analysis folder path {0} is not a folder".format(analysis_folder_path)

        ei_file_path = os.path.join(analysis_folder_path, "{0}.{1}".format(dataset_name, ei_extension))

        assert os.path.isfile(ei_file_path), "EI file {0} does not exist ".format(ei_file_path)

        num_bytes_in_file = os.path.getsize(ei_file_path)

        self.ei_fp = open(ei_file_path, 'rb')

        self.electrode_map = None
        self.disconnected_electrodes = None
        with GlobalsFileReader(analysis_folder_path, dataset_name, globals_extension=globals_extension) as gbfr:
            self.electrode_map, self.disconnected_electrodes = gbfr.get_electrode_map()

        self.n_electrodes = self.electrode_map.shape[0]

        self.nl_points, self.nr_points, self.array_id = struct.unpack('>III', self.ei_fp.read(N_BYTES_32BIT * 3))
        self.n_samples_per_ei = self.nl_points + self.nr_points + 1
        self.header_size = self.ei_fp.tell()

        # each EI on disk consists of pairs of 32 bit numbers, one pair per electrode per sample
        # one number in the pair is the actual EI value, the other number in the pair is the error
        self.num_bytes_per_ei = 2 * N_BYTES_32BIT * self.n_samples_per_ei * (self.n_electrodes + 1)
        # the +1 is necessary because the data for the TTL channel is stored in the EI as well

        self.cell_id_to_offset = {}  # type: Dict[int, int]
        self.cell_id_to_nspikes = {}  # type: Dict[int, int]

        current_read_offset = self.header_size
        num_bytes_stepsize = (self.num_bytes_per_ei + N_BYTES_32BIT * 2)
        while current_read_offset < num_bytes_in_file:
            cell_id_nspikes_as_bytearray = self.ei_fp.read(N_BYTES_32BIT * 2)
            cell_id_as_int, cell_nspikes_as_int = struct.unpack('>II', cell_id_nspikes_as_bytearray)

            self.cell_id_to_nspikes[cell_id_as_int] = cell_nspikes_as_int
            self.cell_id_to_offset[cell_id_as_int] = (current_read_offset + N_BYTES_32BIT * 2)

            current_read_offset += num_bytes_stepsize
            self.ei_fp.seek(current_read_offset, 0)

    def get_all_eis_by_cell_id(self) -> Dict[int, EIContainer]:

        '''
        Gets all of the EIs contained in the file, matches them with cell ids

        Returns:
            dict with format { cell_id (int) : ei_container (EIContainer)}
        '''

        cell_id_to_ei_map = {}
        for cell_id in self.cell_id_to_offset:
            cell_id_to_ei_map[cell_id] = self.get_ei_for_cell_id(cell_id)

        return cell_id_to_ei_map

    def get_ei_for_cell_id(self,
                           cell_id: int) -> EIContainer:

        '''
        Gets the EI for a single cell

        Args:
            cell_id (int) : cell id

        Returns:
            Tuple with type (np.array of float, np.array of float)

            first np array of float with shape (n_electrodes, n_samples) where n_electrodes is the number
                of electrodes, not including the TTL channel contains the EI

            second np array of float with shape (n_electrodes, n_samples) contains the EI error
        '''

        if cell_id not in self.cell_id_to_offset:
            assert False, "cell id {0} not in EI file".format(cell_id)

        full_ei = np.zeros((self.n_electrodes + 1, self.n_samples_per_ei), dtype=np.float32)  # include TTL channel
        ei_error = np.zeros((self.n_electrodes + 1, self.n_samples_per_ei), dtype=np.float32)  # include TTL channel

        # get the raw data for the EI as a bytearray
        self.ei_fp.seek(self.cell_id_to_offset[cell_id], 0)
        ei_raw_data = self.ei_fp.read(self.num_bytes_per_ei)

        vcext.unpack_ei_from_array(ei_raw_data,
                                   self.n_samples_per_ei,
                                   self.n_electrodes + 1,  # include the TTL channel
                                   full_ei,
                                   ei_error)

        # throw away the TTL channel, it is meaningless in the EI
        return EIContainer(full_ei[1:, :],
                           ei_error[1:, :],
                           self.nl_points,
                           self.nr_points,
                           self.n_samples_per_ei,
                           self.cell_id_to_nspikes[cell_id])

    def get_electrode_map(self) -> np.ndarray:
        return self.electrode_map

    def get_disconnected_electrodes(self) -> Set[int]:
        return self.disconnected_electrodes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.ei_fp.close()

    def close(self):
        self.ei_fp.close()


class ParametersFileReader:
    '''
    Class for reading the parameters file

    Note that this class is very tightly coupled with VisionCellDataTable, because both the
        Vision parameters file and VisionCellDataTable represent the data as a large data table
        keyed by specific column names

    It is recommened that you use this class with VisionCellDataTable, otherwise things will
        get really confusing.

    Example usage:

    vcd = VisionCellDataTable()
    with ParametersFileReader('/Volumes/Analysis/2019-01-01-1/data000', 'data000') as pfr:
        pfr.update_visioncelldata_obj(vcd)

    Alternative usage:

    vcd = VisionCellDataTable()
    pfr = ParametersFileReader('/Volumes/Analysis/2019-01-01-1/data000', 'data000')
    pfr.update_visioncelldata_obj(vcd)
    pfr.close()
    '''

    PARAMS_TAG_CONST_WHITENOISE_MOVIE = 130  # type: int
    PARAMS_TAG_CONST_DOUBLE = 3  # type: int
    PARAMS_TAG_CONST_DOUBLE_ARRAY = 4  # type: int
    PARAMS_TAG_CONST_STRING = 5  # type: int

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 parameters_extension: str = 'params') -> None:

        assert os.path.isdir(analysis_folder_path), \
            "Analysis folder path {0} is not a folder".format(analysis_folder_path)

        params_file_path = os.path.join(analysis_folder_path, "{0}.{1}".format(dataset_name, parameters_extension))

        assert os.path.isfile(params_file_path), "Params file {0} does not exist ".format(params_file_path)

        self.params_fp = open(params_file_path, 'rb')

        # figure out the number of columns and the max number of rows
        self.n_cols, self.n_rows, self.n_rows_max = struct.unpack(">III", self.params_fp.read(N_BYTES_32BIT * 3))

        # figure out the names and types of each column in the parameters file
        self.column_names, self.column_types = [], []  # type: List[str], List[str]
        for i in range(self.n_cols):
            colname_len_bytes = struct.unpack(">I", self.params_fp.read(N_BYTES_32BIT))[0]
            self.column_names.append(str(self.params_fp.read(colname_len_bytes).decode('utf-8')))

            coltype_len_bytes = struct.unpack(">I", self.params_fp.read(N_BYTES_32BIT))[0]
            self.column_types.append(str(self.params_fp.read(coltype_len_bytes).decode('utf-8')))

        # parse the seek table to figure out where in the file each (column, row) entry is
        col_row_to_offset = {}  # keys are (col, row), values are offset from the start of the file
        # where the entry at (col, row) is

        col_row_to_dataoffset = {}
        for j in range(self.n_rows):
            for i in range(self.n_cols):
                col_row_to_offset[(i, j)] = struct.unpack(">I", self.params_fp.read(N_BYTES_32BIT))[0]

        # calculate the offset relative to the start of the data section of file
        for j in range(self.n_rows):
            for i in range(self.n_cols):
                col_row_to_dataoffset[(i, j)] = col_row_to_offset[(i, j)] - col_row_to_offset[(0, 0)]

        # load the data section of the file into memory, and then populate the table with appropriate data
        initial_offset_position = col_row_to_offset[(0, 0)]
        self.params_fp.seek(initial_offset_position, 0)

        all_data_array = self.params_fp.read()
        self.col_row_to_arbitrary_data = {}  # type: Dict[Tuple[int, int], Any]

        for j in range(self.n_rows):
            for i in range(self.n_cols):
                offset_position = col_row_to_dataoffset[(i, j)]
                self.col_row_to_arbitrary_data[(i, j)] = self._read_field(all_data_array, offset_position)

    def get_all_field_names(self) -> List[str]:
        return self.column_names

    def _read_field(self,
                    buffer_array: Union[bytes, bytearray],
                    offset_position: int) -> Any:

        idx = offset_position

        temp = struct.unpack('>H', buffer_array[idx:idx + N_BYTES_16BIT])[0]  # 16 bit integer read from
        # the front
        idx += N_BYTES_16BIT

        if temp == 0xFFFF:
            return None

        tag_id = temp >> 6  # top 10 bits
        length = temp & 0x3F  # bottom 6 bits

        if length == 0x3F:
            length = struct.unpack('>H', buffer_array[idx:idx + N_BYTES_16BIT])[0]
            idx += N_BYTES_16BIT
        elif length == 0x3E:
            length = struct.unpack('>I', buffer_array[idx:idx + N_BYTES_32BIT])[0]
            idx += N_BYTES_32BIT

        field_value = None

        if tag_id == ParametersFileReader.PARAMS_TAG_CONST_DOUBLE:
            field_value = struct.unpack('>d', buffer_array[idx:idx + N_BYTES_64BIT])[0]

        elif tag_id == ParametersFileReader.PARAMS_TAG_CONST_DOUBLE_ARRAY:

            n_doubles_in_array = struct.unpack('>I', buffer_array[idx:idx + N_BYTES_32BIT])[0]
            idx += N_BYTES_32BIT

            field_value = np.zeros((n_doubles_in_array,), dtype=np.float64)
            vcext.unpack_64bit_float_from_bytearray(buffer_array,
                                                    n_doubles_in_array,
                                                    idx,
                                                    field_value)

        elif tag_id == ParametersFileReader.PARAMS_TAG_CONST_STRING:
            length_of_string = struct.unpack('>I', buffer_array[idx:idx + N_BYTES_32BIT])[0]
            idx += N_BYTES_32BIT
            field_value = buffer_array[idx:idx + length_of_string].decode('utf-8')
        else:
            assert False, "tag_id {0} in params file is not valid".format(tag_id)

        return field_value

    def update_visioncelldata_obj(self,
                                  visioncelldata_obj: VisionCellDataTable) -> VisionCellDataTable:

        '''
        Updates the entries of a VisionCellDataTable with all of the data loaded
            from the parameters file. Overwrites entries that already exists.

        Note that this code does data type conversions and string manipulations
            to get the correct cell ids and the correct cell class names.

        Args:
            visioncelldata_obj (VisionCellDataTable) : where all of the data
                will be stored.
        '''

        # figure out which column index is the cell id
        # and which column index is the class id
        cell_id_colindex = -1
        class_id_colindex = -1
        for i, col_name in enumerate(self.column_names):
            if col_name == VisionFieldNames.CELLID_FIELDNAME:
                cell_id_colindex = i
            elif col_name == VisionFieldNames.CLASSID_FIELDNAME:
                class_id_colindex = i

        assert cell_id_colindex != -1, "Did not read cell ids from parameter file"
        assert class_id_colindex != -1, "Did no read cell class ids from parameter file"

        for j in range(self.n_rows):

            # handle the special cases where that data has to be modified
            cell_id = int(round(self.col_row_to_arbitrary_data[(cell_id_colindex, j)]))
            visioncelldata_obj.update_data_for_cell_id_and_field_name(cell_id,
                                                                      VisionFieldNames.CELLID_FIELDNAME,
                                                                      cell_id)

            cell_type = self.col_row_to_arbitrary_data[(class_id_colindex, j)]
            if cell_type.startswith('All/'):
                cell_type = cell_type.replace('All/', '', 1)
            if cell_type == 'All':
                cell_type = 'Unclassified'
            cell_type = cell_type.replace('/', ' ')
            cell_type = cell_type.replace('-', ' ')
            visioncelldata_obj.update_data_for_cell_id_and_field_name(cell_id,
                                                                      VisionFieldNames.CLASSID_FIELDNAME,
                                                                      cell_type)

            for i, col_name in enumerate(self.column_names):
                if col_name != VisionFieldNames.CELLID_FIELDNAME and \
                        col_name != VisionFieldNames.CLASSID_FIELDNAME:
                    data_for_field = self.col_row_to_arbitrary_data[(i, j)]
                    visioncelldata_obj.update_data_for_cell_id_and_field_name(cell_id,
                                                                              col_name,
                                                                              data_for_field)

        return visioncelldata_obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.params_fp.close()

    def close(self):
        self.params_fp.close()


'''
container to store the arrays corresponding to STA movie
also stores the stixel size and the refresh time

stixel_size (float)
refresh-time (float)

red (np array of float32, shape (width, height, depth))
red_error (np array of float32, shape (width, height, depth))

etc...
'''
STAContainer = namedtuple('STAContainer',
                          ['stixel_size',
                           'refresh_time',
                           'sta_offset',
                           'red',
                           'red_error',
                           'green',
                           'green_error',
                           'blue',
                           'blue_error'])


class STAReader:
    '''
    Class for reading an STA file

    Example usage:

    with STAReader('/Volumes/Analysis/2019-01-01-1/data000', 'data000') as star:
        stas_by_cell_id = star.get_all_stas_by_cell_id()

    Alternative usage:

    star = STAReader('/Volumes/Analysis/2019-01-01-1/data000', 'data000')
    stas_by_cell_id = star.get_all_stas_by_cell_id()
    star.close()

    '''

    FILE_HEADER_LENGTH_BYTES = 164  # type: int

    # magic number signifies ID is not used
    NEURON_ID_UNUSED = -2147483648  # type: int
    # NEURON_ID_UNUSED = 0x80000000 # some weird stuff going on with 32 bit vs 64 bit integers in Python

    NUM_BYTES_PER_STIXEL = N_BYTES_32BIT * 6  # type: int

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 sta_extension: str = 'sta',
                 read_chunk_size: int = 100) -> None:

        '''
        Reads the header for the STA file and parses the seek table to figure out where
            in the file each STA is stored

        Does not actually load any of the STAs, that is done elsewhere
        '''

        assert os.path.isdir(analysis_folder_path)
        stafile_path = os.path.join(analysis_folder_path, "{0}.{1}".format(dataset_name, sta_extension))
        assert os.path.isfile(stafile_path), "{0} is not a file".format(stafile_path)

        self.sta_fp = open(stafile_path, 'rb')

        # parse the header and the seek table, lazy-load the actual STA data

        # load the first portion of the header
        header_integer_section_bytearray = self.sta_fp.read(N_BYTES_32BIT * 5)
        version, num_entries, height, width, depth = struct.unpack(">iiiii",
                                                                   header_integer_section_bytearray)

        self.version = version  # type: int
        self.num_entries = num_entries  # type: int
        self.width = width  # type: int
        self.height = height  # type: int
        self.depth = depth  # type: int

        # load the second portion of the header
        header_double_int_section_bytearray = \
            self.sta_fp.read((N_BYTES_64BIT * 2) + \
                             N_BYTES_32BIT)
        stixel_size, refresh_time, sta_offset = \
            struct.unpack(">ddi",
                          header_double_int_section_bytearray)
        self.stixel_size = stixel_size  # type: float
        self.refresh_time = refresh_time  # type: float
        self.sta_offset = sta_offset  # type: int

        # jump to the start of the seek table and start parsing that
        self.sta_fp.seek(STAReader.FILE_HEADER_LENGTH_BYTES, 0)

        self.cell_id_to_byte_offset = {}  # type: Dict[int, int]
        for i in range(self.num_entries):
            seek_table_entry_bytearray = self.sta_fp.read(N_BYTES_32BIT + N_BYTES_64BIT)
            cell_id, offset = struct.unpack(">iq", seek_table_entry_bytearray)

            if cell_id != STAReader.NEURON_ID_UNUSED:
                self.cell_id_to_byte_offset[cell_id] = offset

        self.n_bytes_per_sta = STAReader.NUM_BYTES_PER_STIXEL * self.depth * self.height * self.width + \
                               self.depth * (N_BYTES_32BIT * 2 + N_BYTES_64BIT) + N_BYTES_64BIT + N_BYTES_32BIT

        # also generate additional data structures for fast chunked reading of all cells
        byte_offset_to_cell_id = {val: key for key, val in
                                  self.cell_id_to_byte_offset.items()}  # type: Dict[int, int]
        self.sorted_jt_byte_offsets = sorted(list(byte_offset_to_cell_id.keys()))  # type: List[int]
        self.cell_id_in_file_order = [byte_offset_to_cell_id[val] for val in self.sorted_jt_byte_offsets]
        self.n_cells_per_chunk = read_chunk_size

    def chunked_load_all_stas(self) -> Dict[int, STAContainer]:

        n_cells_to_load = len(self.cell_id_in_file_order)

        ret_dict = {}  # type: Dict[int, STAContainer]

        for i in range(0, n_cells_to_load, self.n_cells_per_chunk):
            top_idx = min(i + self.n_cells_per_chunk, n_cells_to_load)
            byte_offsets_to_load = np.array(self.sorted_jt_byte_offsets[i:top_idx])
            offsets_from_chunk_start = list(byte_offsets_to_load - byte_offsets_to_load[0])
            cell_ids_ordered_for_chunk = self.cell_id_in_file_order[i:top_idx]

            n_bytes_to_read = (byte_offsets_to_load[-1] - byte_offsets_to_load[0]) + self.n_bytes_per_sta

            # jump to the first cell in the chunk
            self.sta_fp.seek(byte_offsets_to_load[0], 0)
            chunked_bytes = self.sta_fp.read(n_bytes_to_read)

            # now loop through each chunk
            for cell_id, chunk_offset in zip(cell_ids_ordered_for_chunk, offsets_from_chunk_start):
                chunk_high = chunk_offset + self.n_bytes_per_sta
                ret_dict[cell_id] = self._unpack_single_sta_from_buffer(chunked_bytes[chunk_offset:chunk_high])

        return ret_dict

    def _unpack_single_sta_from_buffer(self,
                                       sta_bytearray: bytes) -> STAContainer:

        # this call replaces vcext.pack_sta_from_bytearray
        rv, re, gv, ge, bv, be = vcppext.unpack_rgb_sta(sta_bytearray,
                                                        self.width,
                                                        self.height,
                                                        self.depth)

        red_value = np.transpose(rv, (1, 2, 0))
        red_error = np.transpose(re, (1, 2, 0))

        green_value = np.transpose(gv, (1, 2, 0))
        green_error = np.transpose(ge, (1, 2, 0))

        blue_value = np.transpose(bv, (1, 2, 0))
        blue_error = np.transpose(be, (1, 2, 0))

        return STAContainer(self.stixel_size,
                            self.refresh_time,
                            self.sta_offset,
                            red_value,
                            red_error,
                            green_value,
                            green_error,
                            blue_value,
                            blue_error)

    def get_sta_for_cell_id(self,
                            cell_id: int) -> STAContainer:
        '''
        Gets the STA for a given cell_id from the file

        Args:
            cell_id (int) : cell_id

        Returns:
            STAContainer, containing info about STA as well as the STA itself
                Each of the color channels and error channels have format
                np.array, dtype=np.float32, with shape (width, height, num_frames)
        '''

        assert cell_id in self.cell_id_to_byte_offset, "Cell id {0} does not have STA".format(cell_id)

        # go to the right point in the file
        self.sta_fp.seek(self.cell_id_to_byte_offset[cell_id], 0)

        # get the info for each STA

        sta_bytearray = self.sta_fp.read(self.n_bytes_per_sta)
        return self._unpack_single_sta_from_buffer(sta_bytearray)

    def get_all_stas_by_cell_id(self) -> Dict[int, STAContainer]:

        '''
        Gets all of the STAs in a file, groups the by cell id

        Returns:
            dict of format {cell_id (int) : STAContainer}
        '''

        cell_id_to_sta_map = {}
        for cell_id in self.cell_id_to_byte_offset:
            cell_id_to_sta_map[cell_id] = self.get_sta_for_cell_id(cell_id)

        return cell_id_to_sta_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sta_fp.close()

    def close(self):
        self.sta_fp.close()


class NeuronsReader:
    '''
    Class for reading .neurons files

    Example usage

    with NeuronsReader('/Volumes/Analysis/9999-99-99-9/data000', 'data000') as nr:
        spike_times_by_cell_id = nr.get_spike_sample_nums_for_all_real_neurons()
    '''

    HEADER_LENGTH_BYTES = (4 * N_BYTES_32BIT) + N_BYTES_64BIT + 128  # type: int
    # the 128 byte section is unused

    TTL_ID = -1  # type: int
    # TTL channel should always have ID -1

    IS_EMPTY = -2147483648  # type: int

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 neuron_extension: str = 'neurons') -> None:
        '''
        Constructor, reads the header and the seek table but doesn't read any of the data
        :param analysis_folder_path: path to analysis folder, i.e. /Volumes/Analysis/9999-99-99-9/data000
        :param dataset_name: dataset name, i.e. data000
        :param neuron_extension:
        '''

        assert os.path.isdir(analysis_folder_path)
        neuronfile_path = os.path.join(analysis_folder_path, "{0}.{1}".format(dataset_name, neuron_extension))
        assert os.path.isfile(neuronfile_path), "{0} is not a file".format(neuronfile_path)

        self.neuron_fp = open(neuronfile_path, 'rb')

        # load the header
        header_beginning = self.neuron_fp.read(N_BYTES_32BIT * 4)
        vers, headcap, nsamples, sample_freq = struct.unpack(">iiii", header_beginning)

        self.version = vers  # type: int
        self.num_slots_in_file = headcap  # type: int
        # number of entries in the seek table, including the ttl channel
        # there is one entry per neuron

        self.n_samples = nsamples  # type: int
        self.sample_freq = sample_freq  # type: int

        # go through the seek table and populate mapping data structure
        # also figure out which electrode the neuron was first identified on
        self.map_neuron_id_to_memory_offset = {}  # type: Dict[int, int]
        self.map_neuron_id_to_initial_electrode = {}  # type: Dict[int, int]

        self.neuron_fp.seek(NeuronsReader.HEADER_LENGTH_BYTES, 0)  # go to the beginning of the
        # seek table section of the file

        # first load the TTL channel data
        seek_entry = self.neuron_fp.read(N_BYTES_32BIT * 2 + N_BYTES_64BIT)
        ttl_id, should_be_zero, seek_offset = struct.unpack(">iiq", seek_entry)
        assert ttl_id == NeuronsReader.TTL_ID, "TTL ID in neurons file is {0}, should be {1}".format(ttl_id,
                                                                                                     NeuronsReader.TTL_ID)
        assert should_be_zero == 0
        self.map_neuron_id_to_memory_offset[NeuronsReader.TTL_ID] = seek_offset

        for i in range(1, self.num_slots_in_file):

            seek_entry = self.neuron_fp.read(N_BYTES_32BIT * 2 + N_BYTES_64BIT)
            neuron_id, identifier_electrode, seek_offset = struct.unpack(">iiq", seek_entry)

            if identifier_electrode == NeuronsReader.IS_EMPTY or identifier_electrode < 0:
                # either the slot is unused or it's empty
                continue
            else:
                self.map_neuron_id_to_memory_offset[neuron_id] = seek_offset
                self.map_neuron_id_to_initial_electrode[neuron_id] = identifier_electrode

    def get_spike_sample_nums_for_neuron(self,
                                         neuron_id: int) -> np.ndarray:
        '''
        Gets the spike times for a single cell id
        :param neuron_id: cell id
        :return: np.ndarray of shape (nspikes, ) corresponding to the sample numbers at which the cell spiked
        '''
        assert neuron_id in self.map_neuron_id_to_memory_offset, "Neuron id {0} not in neurons file".format(neuron_id)

        self.neuron_fp.seek(self.map_neuron_id_to_memory_offset[neuron_id], 0)

        num_spikes_for_neuron = struct.unpack('>i', self.neuron_fp.read(N_BYTES_32BIT))[0]

        spike_times_bytearray = self.neuron_fp.read(N_BYTES_32BIT * num_spikes_for_neuron)
        spike_sample_numbers = np.zeros((num_spikes_for_neuron,), dtype=np.int32)

        vcext.unpack_32bit_integers_from_bytearray(spike_times_bytearray,
                                                   num_spikes_for_neuron,
                                                   spike_sample_numbers)

        return spike_sample_numbers

    def get_TTL_times(self) -> np.ndarray:
        '''
        Gets the TTL times
        :return: np.ndarray of shape (n_ttltimes, )
        '''
        return self.get_spike_sample_nums_for_neuron(NeuronsReader.TTL_ID)

    def get_identifier_electrode_for_neuron(self,
                                            neuron_id: int) -> int:
        '''
        Gets the seed electrode for a given cell id
        :param neuron_id:
        :return: seed electrode number
        '''
        return self.map_neuron_id_to_initial_electrode[neuron_id]

    def get_identifier_electrodes_for_all_real_neurons(self) -> Dict[int, int]:
        '''
        Gets a dictionary containing the seed electrodes for all cells
        :return: dict with format {cell id (int) : seed electrode number (int)}
        '''
        return self.map_neuron_id_to_initial_electrode

    def get_spike_sample_nums_for_all_real_neurons(self) -> Dict[int, np.ndarray]:

        '''
        Gets the spike times for all cells
        :return: dict with format (cell id (int) : spike times (np.ndarray)}
        '''

        id_to_spike_times_mapping = {}
        for neuron_id in self.map_neuron_id_to_memory_offset:
            if neuron_id != NeuronsReader.TTL_ID:
                id_to_spike_times_mapping[neuron_id] = self.get_spike_sample_nums_for_neuron(neuron_id)

        return id_to_spike_times_mapping

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.neuron_fp.close()

    def close(self):
        self.neuron_fp.close()


class SpikesReader:
    '''
    Class for reading a spikes file. The existing documentation on the .spikes
    file format is incomplete, and seemingly inconsistent with the i/o 
    Java functions in the Vision src. As such, some field names are WRONG, and
    other information in the file is not read. Critical information from the
    header is read, and other bytes are skipped such that the only
    information read are the number of spikes, electrode ID, and the actual 
    spike times corresponding to a given electrode.

    Example usage:

    with SpikesReader('/Volumes/Analysis/2019-01-01-1/data000', 
                      'data000') as sr:
        spike_times_by_electrode = sr.get_spiketimes_by_electrode()

    Alternative usage:
    
    sr = SpikesReader('/Volumes/Analysis/2019-01-01-1/data000',
                      'data000')
    spike_times_by_electrode = sr.get_spiketimes_by_electrode()
    sr.close()
    '''

    # Must skip after reading critical information from header to get data.
    N_BYTES_SEEK = 232  # type: int

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 spikes_extension: str = 'spikes') -> None:
        '''
        Constructor. Reads the header from the spikes file, sets some values.
        
        This does not read any of the spike times, yet.
        '''

        assert os.path.isdir(analysis_folder_path)
        spikesfile_path = os.path.join(analysis_folder_path,
                                       "{0}.{1}".format(
                                           dataset_name, spikes_extension))
        assert os.path.isfile(spikesfile_path), \
            "{0} is not a file".format(spikesfile_path)

        # Open spikes file, parse the header, read first 7 values.
        self.spikes_fp = open(spikesfile_path, 'rb')
        header = self.spikes_fp.read((N_BYTES_32BIT * 7))

        # version, array_id, mean_time_constant, threshold will not be set.
        (magic_number,
         version,
         array_id,
         mean_time_constant,
         threshold,
         n_samples,
         sampling_rate) = struct.unpack(">iiiffii", header)

        self.n_samples = n_samples
        self.sampling_rate = sampling_rate

        # Jump ahead to get the number of electrodes (empirically determined).
        self.spikes_fp.seek(SpikesReader.N_BYTES_SEEK, 0)
        n_electrodes_byte_array = self.spikes_fp.read(N_BYTES_32BIT)
        self.n_electrodes = struct.unpack(">i", n_electrodes_byte_array)[0]

    def __check_spike_counts(self, spikes_dict: dict) -> None:
        '''
        Helper function to ensure that all expected spikes for each electrode
        were written, and that the sample times are all valid.

        Don't directly use this method.

        Args:
            spikes_dict: dictionary mapping electrode to various other data.

        Raises a fatal assertion error if anything is invalid.

        '''
        for electrode in np.arange(1, self.n_electrodes):

            # Check the counts, and the values.
            assert spikes_dict[electrode]['spike_cnt'] == \
                   spikes_dict[electrode]['n_spikes'] == \
                   len(spikes_dict[electrode]['spike_times']), \
                ("Electrode's {0} spikes not written "
                 "properly").format(electrode)

            if len(spikes_dict[electrode]['spike_times']) > 0:
                assert min(spikes_dict[electrode]['spike_times']) > 0 \
                       and max(spikes_dict[electrode]['spike_times']) \
                       <= self.n_samples, ("Electrode's {0} spike times are "
                                           "invalid".format(electrode))

    def __get_total_n_spikes(self, spikes_dict: dict) -> int:
        '''
        Helper function to get total number of spikes across all electrodes.

        Don't directly use this method.

        Args:
            spikes_dict: dictionary mapping electrode to various other fields.
        
        Returns:
            Total number of spikes.
        '''
        total_n_spikes = 0

        for electrode in np.arange(1, self.n_electrodes):
            total_n_spikes += spikes_dict[electrode]['n_spikes']

        return total_n_spikes

    def get_spiketimes_by_electrode(self) -> dict:
        '''
        Gets the spike times for all of the electrodes on the array.

        Args:
            N/A

        Returns: 
            SpikesContainer which is essentially a dictionary mapping 
            electrodes to the number of spikes and the spike times found
            (in samples). 
            
            Because of how the byte stream is written, it takes the 
            same amount of time to search for all the spike times of a 
            single electrode as for all the electrodes. While it isn't as
            memory efficient, the default is to cache all the electrode spike
            times. It takes a LONG time to read through this byte stream, 
            so it is reccomended that you write a file with this information 
            instead of making multiple calls to this function.

            NOTE: electrodes here are indexed by 1, to avoid
            confusion with the ttl. This is NOT necessarily true of other 
            classes/methods in this module.
        '''

        '''
        Advance the file pointer to the point where the data are written.
        According to Vision src, after the header, spike times are written
        as follows:
            electrode (short)
            spike time in samples (int)
        '''
        self.spikes_fp.seek(SpikesReader.N_BYTES_SEEK + N_BYTES_64BIT, 0)

        # Initialize dictionary with counters and total spike information. 
        spikes_dict = dict()

        for electrode in np.arange(1, self.n_electrodes):
            n_spikes_byte_array = self.spikes_fp.read(N_BYTES_32BIT)
            spikes_dict[electrode] = dict()
            spikes_dict[electrode]['spike_cnt'] = 0
            spikes_dict[electrode]['spike_times'] = set()
            spikes_dict[electrode]['n_spikes'] = struct.unpack(">i",
                                                               n_spikes_byte_array)[0]

        # Read into a buffer the expected number of bytes, init counters.
        total_n_spikes = self.__get_total_n_spikes(spikes_dict)
        n_spikes_written = 0
        total_bytes = total_n_spikes * (N_BYTES_16BIT + N_BYTES_32BIT)

        buf = self.spikes_fp.read(total_bytes)
        head = 0
        tail = N_BYTES_16BIT + N_BYTES_32BIT
        jump = tail
        print('Loading spike times for %d samples of data ...' % self.n_samples)

        while n_spikes_written < total_n_spikes:
            electrode, spike_time = struct.unpack(">hi", buf[head:tail])
            head += jump
            tail += jump

            # Write to the dictionary, ensuring valid electrode ID.
            spike_cnt = spikes_dict[electrode]['spike_cnt']
            n_spikes = spikes_dict[electrode]['n_spikes']

            if electrode in np.arange(1, self.n_electrodes) and \
                    spike_cnt < n_spikes:
                spikes_dict[electrode]['spike_times'].add(spike_time)
                spikes_dict[electrode]['spike_cnt'] += 1
                n_spikes_written += 1
        print('done.')

        # Check that all the expected spike times were written, then return.
        self.__check_spike_counts(spikes_dict)
        return spikes_dict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.spikes_fp.close()

    def close(self):
        self.spikes_fp.close()


class NoiseReader():
    '''
    Class for reading a noise file. Using a recursive thresholding and RMS
    computation routine, Vision computes the RMS on each channel using the 
    the first several seconds of the recording to compute the RMS on each 
    channel, and writes out .noise files containing that data.

    Example usage:

    with NoiseReader('/Volumes/Analysis/2019-01-01-1/data000',
                     'data000') as rmsr:
        channel_noise = rmsr.get_channel_noise()

    Alternative usage:

    rmsr = NoiseReader('/Volumes/Analysis/2019-01-01-1/data000',
                       'data000')
    channel_noise = rmsr.get_channel_noise()
    rmsr.close()
    '''

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 noise_extension: str = 'noise') -> None:
        '''
        Constructor, intializes object and sets path.
        '''

        assert os.path.isdir(analysis_folder_path)
        noisefile_path = os.path.join(analysis_folder_path,
                                      "{0}.{1}".format(
                                          dataset_name, noise_extension))
        assert os.path.isfile(noisefile_path), \
            "{0} is not a file".format(noisefile_path)
        self.fp = open(noisefile_path, 'r')

    def get_channel_noise(self) -> np.ndarray:
        '''
        Gets the noise on each channel, in RMS (ADC counts). By default,
        Vision writes the noise on the ttl channel, but that's useless, so 
        it is not returned.

        Args:
            N/A

        Returns:
            n_electrode dimensional array of the noise on each channel,
            indexed by electrode.
        '''
        channel_noise = []

        for channel in self.fp:
            channel_noise.append(float(channel.strip()))

        # Don't return the ttl.
        return np.array(channel_noise[1:])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fp.close()

    def close(self):
        self.fp.close()


class ModelReader():
    '''
    Class for reading a model file. Similar to the documentation on the
    .spikes file, the existing file format documentation was apparently 
    outdated ("OUTDATED" written at the top), and so this class, at least
    the implementation, will unlikely read everything written to the file, but
    will return the mapping of electrode to cellid mapping, as displayed in
    CellFinder (i.e. raw neurons).

    Due to the lacking documentation, many of the file jumps are based on 
    reading the src code/reverse engineering unpacking of the bytestream.

    NOTE: electrodes are 1-indexed, because in many contexts, 0 is the ttl.

    NOTE: as of right now, this class fails on .model files generated from
    CellFinder -- there is a strange inconsistency between the .model file
    and the .neurons-raw file, when modified via CellFinder. 
    
    TODO: figure ^ this out.

    Example usage: 

    with ModelReader('/Volumes/Analysis/2019-01-01-1/data000',
                     'data000') as mr:
        raw_neuron_map = mr.get_raw_neurons_by_electrode()

    Alternative usage:

    mr = ModelReader('/Volumes/Analysis/2019-01-01-1/data000','data000')
    raw_neuron_map = mr.get_raw_neurons_by_electrode()
    mr.close()
    '''

    # Constants. 
    SLOT_UNUSED = -1
    SEEK_TABLE_JUMP = 16
    CELL_INDEX_JUMP = 8
    GLOBALS_FILE_EXT = 'globals'
    NEURONS_RAW_FILE_EXT = 'neurons-raw'

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 model_extension: str = 'model') -> None:
        '''
        Constructor. Seeks past the header to set the file pointer. Don't 
        bother reading fields from the header, nothing useful, at least for 
        the i/o in this class. Sets the seek table.

        Because there is no guarantee where the electrodes appear in the 
        byte stream, we must at least construct the seek table for every 
        electrode.
        '''

        assert os.path.isdir(analysis_folder_path)
        modelfile_path = os.path.join(analysis_folder_path,
                                      "{0}.{1}".format(
                                          dataset_name, model_extension))
        assert os.path.isfile(modelfile_path), \
            "{0} is not a file".format(modelfile_path)

        # Store the arguments for later usage.
        self.analysis_folder_path = analysis_folder_path
        self.dataset_name = dataset_name

        # Open the file, and set the file pointer past the header.
        self.model_fp = open(modelfile_path, 'rb')
        header_size = struct.unpack(">i", self.model_fp.read(N_BYTES_32BIT))[0]
        self.slot_loc = header_size + N_BYTES_32BIT
        self.model_fp.seek(self.slot_loc)
        self.n_slots = struct.unpack(">i", self.model_fp.read(N_BYTES_32BIT))[0]

        # Loop through the slots, and set a seek table for each electrode.
        self.seek_table = dict()

        for i in range(self.n_slots):
            electrode_byte_array = self.model_fp.read(N_BYTES_32BIT)
            electrode = struct.unpack(">i", electrode_byte_array)[0]

            if electrode != ModelReader.SLOT_UNUSED:
                self.slot_loc += ModelReader.SEEK_TABLE_JUMP
                self.model_fp.seek(self.slot_loc)
                seek_location_byte_array = self.model_fp.read(N_BYTES_32BIT)
                self.seek_table[electrode] = \
                    struct.unpack(">i", seek_location_byte_array)[0]

    def __get_connected_electrodes(self) -> np.ndarray:
        '''
        Gets the connected electrodes, indexed by 1, which would be written to 
        the model file. In the case of the 512 array, this will be the same as
        the full electrode set.

        Don't directly use this method.
        '''

        # Get electrode map, and construct the full set of electrodes. 
        with GlobalsFileReader(self.analysis_folder_path,
                               self.dataset_name,
                               ModelReader.GLOBALS_FILE_EXT) as gbfr:
            electrode_map, disconnected_electrodes = gbfr.get_electrode_map()

        electrodes = np.arange(1, electrode_map.shape[0] + 1)

        # If no disconnected electrodes, return full set, otherwise remove.
        if not disconnected_electrodes:
            return electrodes

        # Add 1 to the disconnected because method returns 0-indexed. 
        disconnected_electrodes = np.array(list(disconnected_electrodes)) + 1
        return np.setdiff1d(electrodes, disconnected_electrodes)

    def __check_raw_neurons_by_electrode(self) -> tuple:
        '''
        Checks that the electrodes are written as expected, according to the 
        array ID, and that the raw neuron mapping is in accordance with what
        is written to the neurons-raw file. Raises a fatal assert if not.

        Don't directly use this method.
        '''

        # Get connected electrodes, and raww neuron-seed electrode mapping.
        connected_electrodes = self.__get_connected_electrodes()
        nr = NeuronsReader(self.analysis_folder_path,
                           self.dataset_name,
                           ModelReader.NEURONS_RAW_FILE_EXT)
        seed_electrodes_by_neuron = \
            nr.get_identifier_electrodes_for_all_real_neurons()

        # Check electrodes are valid, neurons are valid, and mapping is right.
        for electrode in self.raw_neurons_map:
            assert electrode in connected_electrodes, \
                ".model file electrodes were written wrong to the map."

            for neuron in self.raw_neurons_map[electrode]:
                assert neuron in seed_electrodes_by_neuron.keys(), \
                    ".model file neurons were written wrong to the map."
                assert seed_electrodes_by_neuron[neuron] == electrode, \
                    ".model file neuron-electrode pairing is wrong."

    def get_raw_neurons_by_electrode(self) -> dict:
        '''
        Makes a dictionary mapping electrode ID (1-indexed) to a list of 
        raw neuron IDs.

        It doesn't take very long to do this for each electode, nor is it
        memory expensive, so for now, all the electrodes are returned in a
        dictionary, by default.

        Args: 
            N/A

        Returns:
            Dictionary mapping electrode to raw neuron IDs.
        '''

        # Initialize dictionary, and write the cells, skipping other unknowns.
        self.raw_neurons_map = dict()

        for electrode in self.seek_table:
            self.raw_neurons_map[electrode] = []
            seek_location = self.seek_table[electrode]

            # Skip the first integer (empirically determined ... )
            seek_location += N_BYTES_32BIT

            # Advance file pointer and get n Gaussians.
            self.model_fp.seek(seek_location)
            n_gauss_byte_array = self.model_fp.read(N_BYTES_32BIT)
            n_gauss = struct.unpack(">i", n_gauss_byte_array)[0]

            # For each Gaussian, write the cell ID, skipping the index.
            for i in range(n_gauss):
                seek_location += ModelReader.CELL_INDEX_JUMP
                self.model_fp.seek(seek_location)
                neuron_byte_array = self.model_fp.read(N_BYTES_32BIT)
                neuron_id = struct.unpack(">i", neuron_byte_array)[0]
                self.raw_neurons_map[electrode].append(neuron_id)

        # Check that all the electrodes and cells are correct, and return.
        self.__check_raw_neurons_by_electrode()
        return self.raw_neurons_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model_fp.close()

    def close(self):
        self.model_fp.close()
