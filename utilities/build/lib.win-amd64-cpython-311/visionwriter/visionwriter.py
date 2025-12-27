import struct
import os
from typing import Union, Dict, Sequence, Tuple
from dataclasses import dataclass

import numpy as np

import visionwriter.cython_extensions.visionwrite_cext as vwcext
import visionwriter.visionwrite_cpp_extensions as vwcpp

import bin2py
import electrode_map as el_map

import visionloader as vl

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
