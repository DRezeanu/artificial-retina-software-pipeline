import struct
import os
from typing import Union, Dict, Sequence, Tuple
from dataclasses import dataclass
from typing import Optional

import numpy as np

import visionwriter.cython_extensions.visionwrite_cext as vwcext
import visionwriter.visionwrite_cpp_extensions as vwcpp

import bin2py
import electrode_map as el_map

import visionloader as vl
from tqdm.auto import tqdm

N_BYTES_16BIT = 2
N_BYTES_32BIT = 4
N_BYTES_64BIT = 8


class NeuronsFileWriter:
    '''
    Class for writing Vision .neurons files

    Enables the use of Vision to calculate STAs, EIs even if the
    original data was sorted with an arbitrary spike sorter

    Uses Integer version of .neurons

    Currently does not attempt to save the seed electrode

    Example usage

    with NeuronsFileWriter('/Volumes/Scratch/9999-99-99-9/data000/', 'data000') as nfw:
        nfw.write_neuron_file(spike_times_by_cell_id, ttl_times, n_samples_total)
    '''

    HEADER_LENGTH_BYTES = (4 * N_BYTES_32BIT) + N_BYTES_64BIT + 128
    HEADER_PAD_LENGTH_BYTES = N_BYTES_64BIT + 128
    # the 128 byte section is unused

    SEEK_TABLE_ENTRY_LENGTH_BYTES = 2 * N_BYTES_32BIT + N_BYTES_64BIT

    TTL_ID = -1  # TTL channel should always have ID -1

    TTL_ELECTRODE_DUMMY = 0
    IDENTIFIER_ELECTRODE_DUMMY = 1  # this isn't for Vision, so we won't necessarily have
    # an identifier electrode

    # necessary that this be right for Vision to read the files
    NEURON_FILE_INT_VERSION = 32  # type: int
    NEURON_FILE_SALK_VERSION = 33  # type: int
    NEURON_FILE_DOUBLE_VERSION = 100  # type: int

    def __init__(self,
                 analysis_write_path: str,
                 dset_name: str,
                 neuron_extension: str = 'neurons',
                 sample_freq: int = 20000):

        '''
        Constructor, doesn't actually write anything to disk

        :param analysis_write_path: path to the generated analysis folder, folder must already exist
        :param dset_name: name of the dataset, i.e. data000
        :param neuron_extension: extension of the neurons file
        :param sample_freq: sample rate of the recording system
        '''

        assert os.path.isdir(analysis_write_path)

        neuronfile_writepath = os.path.join(analysis_write_path, "{0}.{1}".format(dset_name, neuron_extension))
        self.neuron_fp = open(neuronfile_writepath, 'wb')

        self.file_version = NeuronsFileWriter.NEURON_FILE_INT_VERSION
        self.sample_freq = int(sample_freq)

    def write_neuron_file(self,
                          spike_times_by_cell_id: Dict[int, np.ndarray],
                          ttl_times: np.ndarray,
                          n_samples_total: int):
        '''
        Writes cell ids and spike times to a neurons file.
        IMPORTANT: Only call this once per neurons file, this implementation requires that
        the entire contents of the neurons file be written all at once.

        :param spike_times_by_cell_id: dictionary mapping integer cell ids to spike times. The spike times must be
            integers, corresponding to the sample number at which the spike occurred. Has format
            {cell id (int) : spike times (np.ndarray)}
        :param ttl_times: np.ndarray of integers corresponding to the sample numbers of each TTL trigger
        :param n_samples_total: total number of samples
        :return: None
        '''

        # first write the header
        num_neurons_to_write = len(spike_times_by_cell_id) + 1  # include ttl channel
        header = struct.pack('>iiii',
                             self.file_version,
                             num_neurons_to_write,
                             n_samples_total,
                             self.sample_freq)
        self.neuron_fp.write(header)

        # now pad the header with the correct number of useless bytes
        padding = bytearray(NeuronsFileWriter.HEADER_PAD_LENGTH_BYTES)
        self.neuron_fp.write(padding)

        # now generate the seek table and write to disk
        seek_table_total_n_bytes = num_neurons_to_write * NeuronsFileWriter.SEEK_TABLE_ENTRY_LENGTH_BYTES

        # first put in the TTL channel
        curr_offset = NeuronsFileWriter.HEADER_LENGTH_BYTES + seek_table_total_n_bytes
        self.neuron_fp.write(struct.pack('>iiq',
                                         NeuronsFileWriter.TTL_ID,
                                         NeuronsFileWriter.TTL_ELECTRODE_DUMMY,
                                         curr_offset))
        curr_offset += ttl_times.shape[0] * N_BYTES_32BIT + N_BYTES_32BIT
        cell_id_list_ordered = sorted(list(spike_times_by_cell_id.keys()))  # sort to deal with bug in load_neurons in
        # the MATLAB codebase
        for cell_id in cell_id_list_ordered:
            spike_times_for_cell_id = spike_times_by_cell_id[cell_id]
            self.neuron_fp.write(struct.pack('>iiq',
                                             cell_id,
                                             NeuronsFileWriter.IDENTIFIER_ELECTRODE_DUMMY,
                                             curr_offset))
            curr_offset += spike_times_for_cell_id.shape[0] * N_BYTES_32BIT + N_BYTES_32BIT

        # now store the TTL channel data
        ttl_times_bytearray = bytearray(N_BYTES_32BIT * ttl_times.shape[0])
        vwcext.pack_32bit_integers_to_bytearray(ttl_times.astype(np.int32), ttl_times_bytearray)
        self.neuron_fp.write(struct.pack('>i', ttl_times.shape[0]))
        self.neuron_fp.write(ttl_times_bytearray)

        for cell_id in cell_id_list_ordered:
            spike_times_for_cell_id = spike_times_by_cell_id[cell_id]
            spike_times_bytearray = bytearray(N_BYTES_32BIT * spike_times_for_cell_id.shape[0])

            vwcext.pack_32bit_integers_to_bytearray(spike_times_for_cell_id.astype(np.int32), spike_times_bytearray)
            self.neuron_fp.write(struct.pack('>i', spike_times_for_cell_id.shape[0]))
            self.neuron_fp.write(spike_times_bytearray)

    def close(self):
        self.neuron_fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.neuron_fp.close()


class GlobalsFileWriter:
    '''
    Class for writing (greatly simplified but valid) globals files
    for interfacing with Vision. Globals files are needed to calculate
    both STAs and EIs

    Compatible with Litke array data, as well as Georges/Daniel's version
    of Vision for arbitrary electrode configurations

    Example usage:

    with GlobalsFileWriter('/Volumes/Scratch/9999-99-99-9/data000', 'data000') as gfw:
        gfw.write_simplified_litke_array_globals_file(1504,
                                                        0,
                                                        0,
                                                        'no comment',
                                                        'no identifier',
                                                        0,
                                                        100000)
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

    VERSION_DEFAULT = 3  # type: int

    GLOBALS_FILE_ID = 237 + (202 << 8) + (168 << 16) + (33 << 24) + (26 << 32) + (125 << 40) + (6 << 48) + (
            101 << 56)  # type: int

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 globals_extension: str = 'globals'):

        '''
        Constructor, does not write anything to disk

        :param analysis_folder_path: path to generated analysis folder, must already exist
        :param dataset_name: name of the dataset, i.e. data000
        :param globals_extension: globals file extension
        '''

        assert os.path.isdir(analysis_folder_path)

        globals_file_writepath = os.path.join(analysis_folder_path, "{0}.{1}".format(dataset_name, globals_extension))
        self.globals_fp = open(globals_file_writepath, 'wb')

    def _write_chunk(self,
                     tag: int,
                     data_bytearray: Union[bytes, bytearray]) -> None:

        '''
        Writes a chunk of data in the Vision GlobalsFile/ChunkFile format

        :param tag: GlobalsFile / ChunkFile tag identifying what the data is
        :param data_bytearray: data to write for the tag
        :return: None
        '''

        num_bytes_data = len(data_bytearray)

        self.globals_fp.write(struct.pack('>iii', tag, 0, num_bytes_data))
        self.globals_fp.write(data_bytearray)

    def _create_dummy_icp_header(self,
                                 packed_array_id: int) -> bytes:

        array_id = packed_array_id & 0xFFFF
        array_part = (packed_array_id >> 16) & 0xFF
        array_n_parts = (packed_array_id >> 24)

        # FIXME currently hardcode fake info
        # so we can use regular Vision for Litke data
        icr = vl.ImageCalibrationParamsReader(1.0,
                                              1.0,
                                              0.0,
                                              0.0,
                                              False,
                                              False,
                                              0.0,
                                              array_id,
                                              array_part,
                                              array_n_parts)

        return icr.to_bytearray()

    def write_simplified_litke_array_globals_file(self,
                                                  array_id: int,
                                                  base_time: int,
                                                  seconds_time: int,
                                                  comment: str,
                                                  dataset_identifier: str,
                                                  dformat: int,
                                                  n_samples: int) -> None:

        '''
        Writes a simplified globals file for Litke 512 or 519 array data

        :param array_id: array id, important that this is correct
        :param base_time:
        :param seconds_time:
        :param comment:
        :param dataset_identifier:
        :param dformat:
        :param n_samples: number of samples in the recording
        :return: None
        '''

        # write the file id
        self.globals_fp.seek(0)
        self.globals_fp.write(struct.pack('>Q', GlobalsFileWriter.GLOBALS_FILE_ID))

        # write the array id as the array id tag
        array_id_as_bytearray = struct.pack('>I', array_id)
        self._write_chunk(GlobalsFileWriter.ARRAYID_TAG, array_id_as_bytearray)
        if el_map.is_litke_512_board(array_id):

            # then write the header as the RDH512 tag
            header = bin2py.PyBinHeader.make_512_header(base_time,
                                                        seconds_time,
                                                        comment,
                                                        dataset_identifier,
                                                        dformat,
                                                        n_samples)

            header_as_bytearray = header.generate_header_in_binary()
            self._write_chunk(GlobalsFileWriter.RDH512_TAG, header_as_bytearray)
            self._write_chunk(GlobalsFileWriter.ICP_TAG, self._create_dummy_icp_header(array_id))

        elif el_map.is_litke_519_board(array_id):
            header = bin2py.PyBinHeader.make_519_header(base_time,
                                                        seconds_time,
                                                        comment,
                                                        dataset_identifier,
                                                        dformat,
                                                        n_samples)
            header_as_bytearray = header.generate_header_in_binary()
            self._write_chunk(GlobalsFileWriter.RDH512_TAG, header_as_bytearray)
            self._write_chunk(GlobalsFileWriter.ICP_TAG, self._create_dummy_icp_header(array_id))
        elif el_map.is_litke_519_board_120(array_id):
            header = bin2py.PyBinHeader.make_519_header_120um(base_time,
                                                              seconds_time,
                                                              comment,
                                                              dataset_identifier,
                                                              dformat,
                                                              n_samples)
            header_as_bytearray = header.generate_header_in_binary()
            self._write_chunk(GlobalsFileWriter.RDH512_TAG, header_as_bytearray)
            self._write_chunk(GlobalsFileWriter.ICP_TAG, self._create_dummy_icp_header(array_id))

        else:
            assert False, 'array id {0} is not a Litke board'.format(array_id)

    def write_simplified_reconfigurable_array_globals_file(self,
                                                           base_time: int,
                                                           seconds_time: int,
                                                           comment: str,
                                                           dataset_identifier: str,
                                                           dformat: int,
                                                           frequency: int,
                                                           n_samples: int,
                                                           electrode_config_ordered_no_ttl: np.ndarray,
                                                           electrode_pitch: float) -> None:
        '''
        Writes simplified globals file for reconfigurable (Hierlemann) array

        :param base_time:
        :param seconds_time:
        :param comment:
        :param dataset_identifier:
        :param dformat:
        :param frequency: sample rate of the recording
        :param n_samples: number of samples in the dataset
        :param electrode_config_ordered_no_ttl: coordinates of the electrodes in order. Shape (n_electrodes, 2)
        :param electrode_pitch: pitch of the electrodes
        :return: None
        '''

        n_electrodes = electrode_config_ordered_no_ttl.shape[0] + 1

        # in this case the array id is known to be 9999
        self.globals_fp.seek(0)
        self.globals_fp.write(struct.pack('>Q', GlobalsFileWriter.GLOBALS_FILE_ID))

        # write the array id as the array id tag
        array_id_as_bytearray = struct.pack('>I', bin2py.FakeArrayID.BOARD_ID_RECONFIGURABLE)
        self._write_chunk(GlobalsFileWriter.ARRAYID_TAG, array_id_as_bytearray)

        header = bin2py.PyBinHeader.make_header_from_parameters(base_time,
                                                                seconds_time,
                                                                comment,
                                                                dataset_identifier,
                                                                dformat,
                                                                bin2py.FakeArrayID.BOARD_ID_RECONFIGURABLE,
                                                                n_electrodes,
                                                                frequency,
                                                                n_samples)

        header_as_bytearray = header.generate_header_in_binary()
        self._write_chunk(GlobalsFileWriter.RDH512_TAG, header_as_bytearray)

        # also need to write the electrode configuration
        # first make up a ludicrously fake TTL channel coordinate
        x_max = np.max(electrode_config_ordered_no_ttl[:, 0])
        y_max = np.max(electrode_config_ordered_no_ttl[:, 1])

        x_min = np.min(electrode_config_ordered_no_ttl[:, 0])
        y_min = np.min(electrode_config_ordered_no_ttl[:, 1])

        fake_x = x_max + (x_max - x_min)
        fake_y = y_max + (y_max - y_min)

        coordinates_with_ttl = np.zeros((n_electrodes, 2))
        coordinates_with_ttl[0, 0] = fake_x
        coordinates_with_ttl[0, 1] = fake_y
        coordinates_with_ttl[1:, :] = electrode_config_ordered_no_ttl

        coordinates_with_ttl_as_32bit_float = coordinates_with_ttl.astype(np.float32)
        n_entries_coords_table = coordinates_with_ttl_as_32bit_float.shape[0] * \
                                 coordinates_with_ttl_as_32bit_float.shape[1]

        coordinates_as_bytearray = bytearray(N_BYTES_32BIT * n_entries_coords_table)
        vwcext.pack_electrode_coordinates_globals(coordinates_with_ttl_as_32bit_float,
                                                  coordinates_as_bytearray)

        self._write_chunk(GlobalsFileWriter.ELECTRODEMAP_TAG, coordinates_as_bytearray)

        # need to write the pitch
        pitch_as_bytearray = struct.pack('>d', electrode_pitch)
        self._write_chunk(GlobalsFileWriter.ELECTRODEPITCH_TAG, pitch_as_bytearray)

    def write_run_time_movie_params (self,
                                     runtime_movie_params : vl.RunTimeMovieParamsReader) \
            -> None:
        rtmp_chunk, mf_chunk = runtime_movie_params.generate_rtmp_in_binary()

        self._write_chunk(GlobalsFileWriter.RTMP_TAG, rtmp_chunk)
        self._write_chunk(GlobalsFileWriter.MONITOR_FREQ_TAG, mf_chunk)

    def close(self):
        self.globals_fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.globals_fp.close()


@dataclass
class WriteableEIData:
    ei_matrix: np.ndarray
    ei_error: np.ndarray
    n_spikes: int


class EIWriter:

    HEADER_SIZE = 3 * N_BYTES_32BIT

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 left_samples: int,
                 right_samples: int,
                 array_id: int,
                 overwrite_existing: bool = False,
                 ei_extension: str = 'ei') -> None:

        if not os.path.isdir(analysis_folder_path):
            raise FileNotFoundError('{0} does not exist'.format(analysis_folder_path))

        ei_file_path = os.path.join(analysis_folder_path, "{0}.{1}".format(dataset_name, ei_extension))

        # safety checks to make sure that the caller is OK with overwriting existing
        if os.path.isfile(ei_file_path) and not overwrite_existing:
            raise FileExistsError('{0} already exists, and EIWriter overwrite_existing=False'.format(ei_file_path))

        self.left_samples, self.right_samples, self.array_id = left_samples, right_samples, array_id
        self.samples_per_ei = left_samples + right_samples + 1

        self.ei_fp = open(ei_file_path, 'wb')

        # generate and write the header
        header = struct.pack('>III', left_samples, right_samples, array_id)
        self.ei_fp.write(header)

    def write_eis_by_cell_id(self,
                             writeable_ei_by_cell_id: Dict[int, WriteableEIData]) -> None:
        '''

        :param eis_by_cell_id:
        :return:
        '''

        self.ei_fp.seek(EIWriter.HEADER_SIZE, os.SEEK_SET)
        for ei_cell_id, writeable_ei in writeable_ei_by_cell_id.items():

            packed_id_spikes = struct.pack('>II', ei_cell_id, writeable_ei.n_spikes)
            self.ei_fp.write(packed_id_spikes)

            # now pack the np.ndarray into a matrix
            packed_ei_data = vwcpp.pack_ei_matrices(writeable_ei.ei_matrix,
                                                    writeable_ei.ei_error)

            self.ei_fp.write(packed_ei_data)

    def close(self):
        self.ei_fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.ei_fp.close()




class STAWriter:
    FILE_HEADER_LENGTH_BYTES = 164  # type: int

    BASIC_VERSION = 32  # type: int

    # magic number signifies ID is not used
    NEURON_ID_UNUSED = -2147483648  # type: int

    # NEURON_ID_UNUSED = 0x80000000 # some weird stuff going on with 32 bit vs 64 bit integers in Python

    def __init__(self,
                 analysis_folder_path: str,
                 dataset_name: str,
                 width: int,
                 height: int,
                 depth: int,
                 refresh_time: float,
                 sta_offset: int,
                 stixel_size: float,
                 sta_extension: str = 'sta') -> None:
        assert os.path.isdir(analysis_folder_path), \
            "analysis folder path must exist"

        sta_writepath = os.path.join(analysis_folder_path, "{0}.{1}".format(dataset_name, sta_extension))
        self.sta_fp = open(sta_writepath, 'wb')

        self.n_bytes_written = 0  # type: int

        self.width = width
        self.height = height
        self.depth = depth
        self.refresh_time = refresh_time
        self.sta_offset = sta_offset
        self.stixel_size = stixel_size

    def _write_header(self,
                      num_entries: int) -> None:
        assert self.n_bytes_written == 0, 'header must be written at the beginning of the STA file'

        integer_section = struct.pack('>iiiii',
                                      STAWriter.BASIC_VERSION,
                                      num_entries,
                                      self.height,
                                      self.width,
                                      self.depth)
        self.sta_fp.write(integer_section)
        self.n_bytes_written += 5 * N_BYTES_32BIT  # 5 * 4 = 20

        float_section = struct.pack('>ddi', self.stixel_size, self.refresh_time, self.sta_offset)
        self.sta_fp.write(float_section)
        self.n_bytes_written += 2 * N_BYTES_64BIT + N_BYTES_32BIT  # 2 * 8 + 4 = 20

        self.entry_size_bytes = 6 * self.width * self.height * self.depth * N_BYTES_32BIT + self.depth * (
                    N_BYTES_32BIT * 2 + N_BYTES_64BIT) + N_BYTES_32BIT + N_BYTES_64BIT

        # now write a bunch of zeros until we get to 164 bytes total
        # this is exactly 124 bytes
        self.sta_fp.write(bytearray(124))
        self.n_bytes_written += 124

    def _write_single_entry_buffer_contents(self,
                                            sta_cont: vl.STAContainer) \
            -> None:
        r_data_swap = np.transpose(sta_cont.red, (2, 0, 1))
        g_data_swap = np.transpose(sta_cont.green, (2, 0, 1))
        b_data_swap = np.transpose(sta_cont.blue, (2, 0, 1))

        r_err_swap = np.transpose(sta_cont.red_error, (2, 0, 1))
        g_err_swap = np.transpose(sta_cont.green_error, (2, 0, 1))
        b_err_swap = np.transpose(sta_cont.blue_error, (2, 0, 1))

        output_buffer = vwcpp.pack_sta_buffer_color(r_data_swap,
                                                    r_err_swap,
                                                    g_data_swap,
                                                    g_err_swap,
                                                    b_data_swap,
                                                    b_err_swap,
                                                    self.refresh_time)
        self.sta_fp.write(struct.pack('>di', self.refresh_time, self.depth))
        self.sta_fp.write(output_buffer)

        self.n_bytes_written += self.entry_size_bytes

    def _write_jump_table(self,
                          cell_id_order: Sequence[int]):

        # we must write the jump table in a specific place
        assert self.n_bytes_written == STAWriter.FILE_HEADER_LENGTH_BYTES, \
            'jump table must be written immediately after the header'

        write_position = STAWriter.FILE_HEADER_LENGTH_BYTES + len(cell_id_order) * (N_BYTES_32BIT + N_BYTES_64BIT)
        # note that we have to account for the size of the jump table as well

        for cell_id in cell_id_order:
            jt_entry = struct.pack('>iq', cell_id, write_position)
            self.sta_fp.write(jt_entry)
            self.n_bytes_written += (N_BYTES_64BIT + N_BYTES_32BIT)

            write_position += self.entry_size_bytes

    def write_sta_by_cell_id(self,
                             sta_by_cell_id: Dict[int, vl.STAContainer]) \
            -> None:
        # we have to be very careful about the order in which we write things
        # has to be consistent with the jump table

        cell_order = list(sta_by_cell_id.keys())
        self._write_header(len(cell_order))
        self._write_jump_table(cell_order)

        for cell_id in cell_order:
            sta_cont = sta_by_cell_id[cell_id]
            self._write_single_entry_buffer_contents(sta_cont)

    def close(self):
        self.sta_fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sta_fp.close()


class STAWriterNP(object):
    HEADER_LENGTH_BYTES = (7 * N_BYTES_32BIT) + N_BYTES_64BIT + 128
    HEADER_PAD_LENGTH_BYTES = N_BYTES_64BIT + 128

    def __init__(self, filepath):
        """
        Constructor method
        Parameters:
            filepath: Full path to the .sta file that you intend to write
        """
        self.filepath = filepath
        self.version = 32
        self.max_neurons = 10000
        self.fp = None

    def write(
        self,
        sta: np.ndarray,
        ste: Optional[np.ndarray],
        cluster_id: np.ndarray,
        stixel_size: float=30.0,
        frame_refresh: float=1000/120,
    ) -> None:
        """
        Arguments:
            sta: Tensor containing the spike-triggered average. Dimensions:[cell,time,y,x,RGB]  (type: numpy.ndarray)
            cluster_id: Array containing the cluster/cell ids. Dimensions:[cell] (type: numpy.ndarray)
            stixel_size: Scalar value for the edge length of each stixel in microns (type: float)
            frame_refresh: Scalar value for the temporal refresh of the STA in milliseconds (type: float)
        """
        n_cells = sta.shape[0] # Number of cells
        sta_width = sta.shape[3] # Width
        sta_height = sta.shape[2] # Height
        sta_depth = sta.shape[1] # Time depth
        sta_colors = sta.shape[4] # Number of color channels for STA

        if ste is None:
            ste = np.zeros_like(sta)

        # Reverse the STA time dimension for vision.
        sta = sta[:,::-1,:,:,:]

        # Rescale the STA from contrast to 0-1 for Vision, if needed.
        if np.min(sta) > -0.001:
            sta = 0.5 * sta + 0.5

        # Calculate the size in bytes of a single cell's STA.
        sta_size = 12 + sta_depth * ((sta_width * sta_height * sta_colors + 2) * 8)

        # Get the first cell's location.
        first_location = n_cells * 4 + n_cells * 8 + 164
        
        # Open the file for writing.
        with open(self.filepath, 'wb') as fp:
            # Write the header
            fp.write( struct.pack('>IIIII', self.version, n_cells, sta_width, sta_height, sta_depth) )

            # Write the stixel size and frame refresh to the header.
            fp.write( struct.pack('>d', stixel_size) )
            fp.write( struct.pack('>d', frame_refresh) )

            # Skip the header (164 bytes) and write the cluster_ids and data locations.
            fp.seek(STAWriterNP.HEADER_LENGTH_BYTES, os.SEEK_SET) # fp.seek(164, os.SEEK_SET)

            # Write the Cell Ids and data locations
            for i, id in enumerate(cluster_id):
                fp.write( struct.pack('>I', id) ) # Cluster ID or Cell Number
                fp.write( struct.pack('>Q', sta_size * i + first_location) ) # Data locations

            # Write each STA.
            for cell in tqdm(range(n_cells), desc ="Writing STA file ..."):
                # Write frame refresh and depth for each STA.
                fp.write( struct.pack('>d', frame_refresh) )
                fp.write( struct.pack('>I', sta_depth) )
                output_buffer = vwcpp.pack_sta_buffer_color(sta[cell,:,:,:,0],
                                                            ste[cell,:,:,:,0],
                                                            sta[cell,:,:,:,1],
                                                            ste[cell,:,:,:,1],
                                                            sta[cell,:,:,:,2],
                                                            ste[cell,:,:,:,2],
                                                            stixel_size)
                fp.write(output_buffer)
                # for t in range(sta_depth):
                    # Write the width, height, and stixel size for each frame.
                    # fp.write( struct.pack('>II', sta_width, sta_height) )
                    # fp.write( struct.pack('>d', stixel_size) )

                    # output_buffer = utilcpp.pack_sta_buffer(sta[cell,t,:,:,0],ste[cell,t,:,:,0],sta[cell,t,:,:,1],ste[cell,t,:,:,1],sta[cell,t,:,:,2],ste[cell,t,:,:,2], stixel_size)
                    # fp.write(output_buffer)
                    
                    # for y in range(sta_height):
                    #     for x in range(sta_width):
                    #         for color in range(sta_colors):
                    #             fp.write( struct.pack('>f', sta[cell,t,y,x,color]) ) # STA value
                    #             if ste is None:
                    #                 fp.write( struct.pack('>f', 0.0) ) # Error value
                    #             else:
                    #                 fp.write( struct.pack('>f', ste[cell,t,y,x,color]) ) # STE/STV value
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fp is not None:
            self.fp.close()

    def close(self):
        if self.fp is not None:
            self.fp.close()


class ParamsWriter(object):
    PARAM_NAMES_AND_TYPES = {
        'ID': 'Double', 'classID': 'String', 'x0': 'Double', 'y0': 'Double', 'SigmaX': 'Double', 'SigmaY': 'Double', 
        'Theta': 'Double', 'gAmp': 'Double', 'contourX': 'DoubleArray', 'contourY': 'DoubleArray', 'contourArea': 'Double', 
        'simpleContourX': 'DoubleArray', 'simpleContourY': 'DoubleArray', 'simpleContourArea': 'Double', 'EIx0': 'Double', 
        'EIy0': 'Double', 'EISigmaX': 'Double', 'EISigmaY': 'Double', 'EITheta': 'Double', 'EIgAmp': 'Double', 
        'RedTimeCourse': 'DoubleArray', 'GreenTimeCourse': 'DoubleArray', 'BlueTimeCourse': 'DoubleArray', 't1': 'Double', 
        't2': 'Double', 't3': 'Double', 'a1': 'Double', 'a2': 'Double', 'a3': 'Double', 'n1': 'Double', 'n2': 'Double', 
        'n3': 'Double', 'tOffset': 'Double', 'dot': 'Double', 'dot2': 'Double', 'srm': 'Double', 'rl': 'Double', 
        'amp1': 'Double', 'amp2': 'Double', 'amp3': 'Double', 'blueness': 'Double', 'RedVTimeCourse': 'DoubleArray', 
        'GreenVTimeCourse': 'DoubleArray', 'BlueVTimeCourse': 'DoubleArray', 't1V': 'Double', 't2V': 'Double', 
        't3V': 'Double', 'a1V': 'Double', 'a2V': 'Double', 'a3V': 'Double', 'n1V': 'Double', 'n2V': 'Double', 
        'n3V': 'Double', 'tVOffset': 'Double', 'dotV': 'Double', 'dot2V': 'Double', 'srmV': 'Double', 'rlV': 'Double', 
        'amp1V': 'Double', 'amp2V': 'Double', 'amp3V': 'Double', 'bluenessV': 'Double', 'Auto': 'DoubleArray', 
        'acfBinning': 'Double', 'nSpikes': 'Double', 'acfMean': 'Double', 'acfRMS': 'Double', 'contamination': 'Double',
        'isiBinning': 'Double', 'isiMean': 'Double', 'isiRMS': 'Double'}
    DOUBLE_SIZE = 10
    STRING_SIZE = 9
    
    def __init__(self, filepath: str, cluster_id: np.ndarray):
        """
        Constructor method
        Parameters:
            filepath: Full path to the .params file that you intend to write
            cluster_id: List of cluster ids that align with those contained in the .neurons file
        """

        self.filepath = filepath
        self.cluster_id = cluster_id
        # Parameter names and data types.
        self.param_names_and_types = self.getParamDict()
        self.data_sizes = [] # [10, 9, 10, 10, 10, 10, 10, 10, 6, 6, 10, 6, 6, 10, 408, 408, 408, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6, 6, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1608, 10, 10, 10, 10]
        self.fp = open(filepath, 'wb')

    def check_timecourse(self, time_course: Optional[np.ndarray]):
        if time_course is not None:
            num_timecourse = time_course.shape[1]
        else:
            num_timecourse = 0
        if num_timecourse == 0:
            time_course = None
        return time_course, num_timecourse

    def check_contour_data(self, contour_data: Optional[np.ndarray]):
        if contour_data is not None:
            contour_counts = np.count_nonzero(contour_data[:,:,0], axis=1)
            num_contours = np.max(contour_counts)
        else:   
            num_contours = 0
        if num_contours == 0:
            contour_data = None
        return contour_data, num_contours

    def write_header(self, num_params: int, num_cells: int, maxNeurons: int=10000):
        """ Write the header of the .params file. 
        Arguments:
            num_params: Number of parameters.
            num_cells: Number of cells.
            maxNeurons: Maximum number of neurons.
        """
        # Write the header
        print('Writing file header.')
        self.fp.write( struct.pack('>III', num_params, num_cells, maxNeurons) )
    
    def write_parameter_dict(self):
        """ Write the parameter names and data types."""
        for param in self.param_names_and_types:
            for i in range(3):
                self.fp.write(struct.pack('>b', 0))
            l = len(param)
            self.fp.write(struct.pack('>b', l))
            for k in range(l):
                self.fp.write(struct.pack('>s', bytes(param[k], 'utf-8')))
            for i in range(3):
                self.fp.write(struct.pack('>b', 0))
            value = self.param_names_and_types[param]
            l = len(value)
            self.fp.write(struct.pack('>b', l))
            for k in range(l):
                self.fp.write(struct.pack('>s', bytes(value[k], 'utf-8')))
    
    def write_file_locations(self, locs: list):
        """ Write the file locations for each parameter. 
        Arguments:
            locs: List of file locations for each parameter.
        """
        for loc in locs:
            self.fp.write(struct.pack('>I', loc-2))

        end_of_fid = locs[-1]+8
        num = end_of_fid - self.fp.tell()
        for i in range(num):
            self.fp.write(struct.pack('>b', 0))
    
    def write(
        self,
        timecourse_matrix,
        isi,
        spike_count,
        x0,
        y0,
        sigma_x,
        sigma_y,
        theta: Optional[np.ndarray]=None,
        isi_binning: float=0.5, 
        contour_xy: Optional[np.ndarray]=None,
        contour_area: Optional[np.ndarray]=None,
        simple_contour_xy: Optional[np.ndarray]=None,
        simple_contour_area: Optional[np.ndarray]=None,
        timecourse_variance: Optional[np.ndarray]=None,
        ei_parameters: Optional[np.ndarray]=None,
    ):
        """
        Arguments:
            timecourse_matrix: Timecourse matrix/temporal filters. Dimensions:[cell,time,RGB]  (type: numpy.ndarray)
            isi: Interspike interval distributions. Dimensions:[cell, time] (type: numpy.ndarray)
            spike_count: Total spike counts to noise stimulus. Dimensions:[cell] (type: numpy.ndarray)
            x0: X-location (stixels) of RF center (type: numpy.ndarray)
            y0: Y-location (stixels) of RF center (type: numpy.ndarray)
            sigma_x: X-standard deviation of RF Gaussian (type: numpy.ndarray)
            sigma_y: Y-standard deviation of RF Gaussian (type: numpy.ndarray)
            theta: Rotational angle of Gaussian RF (in degrees) (type: numpy.ndarray)
            isi_binning: Bin width (in milliseconds) used for computing the ISI distribution (type: float, default: 0.5)
            hull_vertices: Vertices of the convex hull of the RF (type: numpy.ndarray)
            hull_area: Area of the convex hull of the RF (type: numpy.ndarray)
        """
        
        num_cells = timecourse_matrix.shape[0]
        maxNeurons = 10000
        cellNameString = [0, 0, 0, 3, 65, 108, 108]

        # Set the data sizes based on the inputs.
        num_timecourse = timecourse_matrix.shape[1]
        num_isi = isi.shape[1]
        # Check inputs to make sure they are valid.
        timecourse_variance, num_v_timecourse = self.check_timecourse(timecourse_variance)
        contour_xy, num_contours = self.check_contour_data(contour_xy)
        simple_contour_xy, num_simple_contours = self.check_contour_data(simple_contour_xy)
        # Remove the irrelevant params and get the data lengths.
        self.set_data_sizes(
            num_timecourse=num_timecourse,
            num_isi=num_isi,
            num_v_timecourse=num_v_timecourse,
            num_contours=num_contours,
            num_simple_contours=num_simple_contours
        )
        
        # Get the number of relevant parameters.
        num_params = len(self.param_names_and_types)

        # Write the header
        self.write_header(
            num_params=num_params,
            num_cells=num_cells,
            maxNeurons=maxNeurons,
        )

        # Write the parameter names and data types.
        self.write_parameter_dict()

        # Get the data locations.
        locs = self.get_data_locations(num_params, num_cells)

        # Write the locations.
        self.write_file_locations(locs)

        # Write the parameter values for each cell.
        param_count = 0
        for cellID in tqdm(range(num_cells), desc='Writing parameters for each cell'):
            for i in range(num_params):
                self.fp.seek(locs[param_count]-2, os.SEEK_SET)
                param_count += 1
                param_key = list(self.param_names_and_types.keys())[i]
                param_value = list(self.param_names_and_types.values())[i]
                if param_value == 'String':
                    self.fp.write(struct.pack('>h', 327))
                    for v in cellNameString:
                        self.fp.write(struct.pack('>b', v))
                elif list(self.param_names_and_types.values())[i] == 'DoubleArray':
                    is_array = False
                    if (param_key == 'RedTimeCourse' or param_key == 'GreenTimeCourse' or param_key == 'BlueTimeCourse'):
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 404))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_timecourse))
                        is_array = True
                    elif param_key == 'Auto':
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 1604))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_isi))
                        is_array = True
                    elif (timecourse_variance is not None) and (param_key == 'RedVTimeCourse' or param_key == 'GreenVTimeCourse' or param_key == 'BlueVTimeCourse'):
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 404))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_v_timecourse))
                        is_array = True
                    elif (contour_xy is not None) and (param_key == 'contourX' or param_key == 'contourY'):
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 404))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_contours))
                        is_array = True
                    elif (simple_contour_xy is not None) and (param_key == 'simpleContourX' or param_key == 'simpleContourY'):
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 404))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_simple_contours))
                        is_array = True
                    else:
                        self.fp.write(struct.pack('>h', 260))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', 0))
                    
                    if is_array:
                        if param_key == 'RedTimeCourse':
                            my_array = timecourse_matrix[cellID, :, 0]
                        elif param_key == 'GreenTimeCourse':
                            my_array = timecourse_matrix[cellID, :, 1]
                        elif param_key == 'BlueTimeCourse':
                            my_array = timecourse_matrix[cellID, :, 2]
                        elif param_key == 'RedVTimeCourse':
                            my_array = timecourse_variance[cellID, :, 0]
                        elif param_key == 'GreenVTimeCourse':
                            my_array = timecourse_variance[cellID, :, 1]
                        elif param_key == 'BlueVTimeCourse':
                            my_array = timecourse_variance[cellID, :, 2]
                        elif param_key == 'contourX':
                            my_array = contour_xy[cellID, :num_contours, 0]
                        elif param_key == 'contourY':
                            my_array = contour_xy[cellID, :num_contours, 1]
                        elif param_key == 'simpleContourX':
                            my_array = simple_contour_xy[cellID, :num_simple_contours, 0]
                        elif param_key == 'simpleContourY':
                            my_array = simple_contour_xy[cellID, :num_simple_contours, 1]
                        else:
                            my_array = isi[cellID, :]
                        for val in my_array:
                            self.fp.write(struct.pack('>d', val))
                else:
                    if param_key == 'ID':
                        k = self.cluster_id[cellID] #cellID + 1
                    elif param_key == 'x0':
                        k = np.max([1.0, x0[cellID]])
                    elif param_key == 'y0':
                        k = np.max([1.0, y0[cellID]])
                    elif param_key == 'SigmaX':
                        k = np.max([1.0, sigma_x[cellID]])
                    elif param_key == 'SigmaY':
                        k = np.max([1.0, sigma_y[cellID]])
                    elif param_key == 'Theta':
                        if theta is None:
                            k = 0.0
                        else:
                            k = theta[cellID]
                    elif param_key == 'contourArea':
                        if contour_area is not None:
                            k = contour_area[cellID]
                        else:
                            k = 0.0
                    elif param_key == 'simpleContourArea':
                        if simple_contour_area is not None:
                            k = simple_contour_area[cellID]
                        else:
                            k = 0.0
                    elif param_key == 'nSpikes':
                        k = spike_count[cellID]
                    elif param_key == 'tOffset':
                        k = 0.0
                    elif (param_key == 'acfBinning' or param_key == 'isiBinning'):
                        k = isi_binning
                    elif (param_key == 'acfMean' or param_key == 'isiMean'):
                        k = np.mean(isi[cellID, :])
                    elif (param_key == 'acfRMS' or param_key == 'isiRMS'):
                        k = 43.1999
                    elif param_key == 'EIgAmp':
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 0]
                        else:
                            k = 0.0
                    elif (param_key == 'EIx0'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 1]
                        else:
                            k = 0.0
                    elif (param_key == 'EIy0'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 2]
                        else:
                            k = 0.0
                    elif (param_key == 'EISigmaX'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 3]
                        else:
                            k = 0.0
                    elif (param_key == 'EISigmaY'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 4]
                        else:
                            k = 0.0
                    elif (param_key == 'EITheta'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 5]
                        else:
                            k = 0.0
                    elif (param_key == 'blueness' or param_key == 'amp1' or param_key == 'amp2' or param_key == 'amp3'):
                        k = 0.0
                    else:
                        k = 0.0
                    
                    self.fp.write(struct.pack('>h', 200))
                    self.fp.write(struct.pack('>d', k))
    
    def getParamDict(self):
        # param_keys =  ['ID', 'classID', 'x0', 'y0', 'SigmaX', 'SigmaY', 'Theta', 'gAmp', 'contourX', 'contourY', 'contourArea', 'simpleContourX', 'simpleContourY', 'simpleContourArea', 'RedTimeCourse', 'GreenTimeCourse', 'BlueTimeCourse', 't1', 't2', 't3', 'a1', 'a2', 'a3', 'n1', 'n2', 'n3', 'tOffset', 'dot', 'dot2', 'srm', 'rl', 'amp1', 'amp2', 'amp3', 'blueness', 'RedVTimeCourse', 'GreenVTimeCourse', 'BlueVTimeCourse', 't1V', 't2V', 't3V', 'a1V', 'a2V', 'a3V', 'n1V', 'n2V', 'n3V', 'tVOffset', 'dotV', 'dot2V', 'srmV', 'rlV', 'amp1V', 'amp2V', 'amp3V', 'bluenessV', 'Auto', 'isiBinning', 'nSpikes', 'isiMean', 'isiRMS']
        # param_values = ['Double', 'String', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'DoubleArray', 'Double', 'DoubleArray', 'DoubleArray', 'Double', 'DoubleArray', 'DoubleArray', 'DoubleArray', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'DoubleArray', 'DoubleArray', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'Double', 'Double', 'Double', 'Double']

        # paramNames = dict()
        # for i, key in enumerate(param_keys):
        #     paramNames[key] = param_values[i]
        paramNames = ParamsWriter.PARAM_NAMES_AND_TYPES.copy()
        return paramNames

    def get_data_locations(self, num_params: int, num_cells: int):
        """ Get the data locations for each parameter. 
        Arguments:
            num_params: Number of parameters.
            num_cells: Number of cells. 
        Returns:
            locs: List of data locations for each parameter.
        """
        locs = []
        locs.append(2441282)
        count = 0
        for cell in range(num_cells):
            for param in range(num_params):
                locs.append(locs[count] + self.data_sizes[param])
                count += 1
        locs = locs[:-1]
        return locs
    
    def set_data_sizes(self, num_timecourse: int=61, num_isi: int=300, num_v_timecourse: int=0, num_contours: int=0, num_simple_contours: int=0):
        """ Set the data sizes for each parameter. 
        Arguments:
            num_timecourse: Number of timecourse points.
            num_isi: Number of isi points.
            num_v_timecourse: Number of variance timecourse points.
            num_contours: Number of contour points.
            num_simple_contours: Number of simple contour points.
        """
        self.data_sizes = list()

        for key, value in self.param_names_and_types.items():
            if value == 'Double':
                self.data_sizes.append(ParamsWriter.DOUBLE_SIZE)
            elif value == 'String':
                self.data_sizes.append(ParamsWriter.STRING_SIZE)
            elif value == 'DoubleArray':
                if (key == 'RedTimeCourse' or key == 'GreenTimeCourse' or key == 'BlueTimeCourse'):
                    self.data_sizes.append((num_timecourse+1) * N_BYTES_64BIT)
                elif key == 'Auto':
                    self.data_sizes.append((num_isi+1) * N_BYTES_64BIT)
                elif (num_v_timecourse > 0) and (key == 'RedVTimeCourse' or key == 'GreenVTimeCourse' or key == 'BlueVTimeCourse'):
                    self.data_sizes.append((num_v_timecourse+1) * N_BYTES_64BIT)
                elif (num_contours > 0) and (key == 'contourX' or key == 'contourY'):
                    self.data_sizes.append((num_contours+1) * N_BYTES_64BIT)
                elif (num_simple_contours > 0) and (key == 'simpleContourX' or key == 'simpleContourY'):
                    self.data_sizes.append((num_simple_contours+1) * N_BYTES_64BIT)
                else:
                    self.data_sizes.append(6)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fp is not None:
            self.fp.close()

    def close(self):
        if self.fp is not None:
            self.fp.close()
