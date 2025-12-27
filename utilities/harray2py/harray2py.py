import h5py
import numpy as np

from typing import Tuple, Dict, List, Optional, Union

import os

class HArrayDataReader:

    TRIGGER_KEY = 'bits'
    DATA_KEY = 'sig'
    MAPPING_KEY = 'mapping'
    SETTINGS_KEY = 'settings'

    GAIN_SUBKEY = 'gain'
    LSB_SUBKEY = 'lsb'
    CUTOFF_FREQ_SUBKEY = 'hpf'

    MAX_DATA_CHANNELS = 1028
    MAX_AMPLIFIER_CHANNELS = 1024

    def __init__(self,
                 file_path: str):

        self.file_handle = h5py.File(file_path, 'r')

        raw_data = self.file_handle[HArrayDataReader.DATA_KEY]
        self.first_frame_num = ((raw_data[1027, 0] << 16) | raw_data[1026, 0])  # type: int

        # automatically determine the electrode map, electrode id, and channel selector mask
        self.electrode_boolean_selector = np.zeros((HArrayDataReader.MAX_DATA_CHANNELS,), dtype=np.bool)
        self.connected_electrodes_list = []
        electrode_mapping_info = self.file_handle[HArrayDataReader.MAPPING_KEY]
        # parse the mapping
        coordinates_list = []  # type: List[Tuple[float]]
        for i, mapping_row in enumerate(electrode_mapping_info):
            amp_id, el_id, xpos, ypos = mapping_row
            self.electrode_boolean_selector[amp_id] = True
            self.connected_electrodes_list.append(el_id)

        self.electrode_map = np.array(coordinates_list)

    def get_data(self, start_sample: int, n_samples: int) -> np.ndarray:
        '''
        Gets raw unfiltered data from Hierlemann array hdf5 file

        Automatically excludes channels that are not recording anything

        Programming notes: there are 1028 channels in the data section
        The first 1024 channels potentially contain recorded data from
            the electrodes, depending on whether the amplifier is connected
            or not

        Channels 1024-1027 are more complicated. Channels 1026 and 1027 must
            be combined to get the frame number of a particular sample

        :param start_sample: start sample index, integer
        :param n_samples: number of samples to get, starting at start_sample
        :return: np.ndarray, shape (n_channels, time)
        '''

        raw_data = self.file_handle[HArrayDataReader.DATA_KEY]
        data_chunk_all_channels = raw_data[:, start_sample:start_sample + n_samples]
        return np.ascontiguousarray(data_chunk_all_channels[self.electrode_boolean_selector, :])

    def get_connected_electrode_mask(self) -> np.ndarray:
        return np.copy(self.electrode_boolean_selector)

    def get_connected_electrode_id(self) -> List[int]:
        return [x for x in self.connected_electrodes_list]

    def get_electrode_map(self) -> np.ndarray:
        return np.copy(self.electrode_map)

    def get_trigger_times(self) -> Optional[np.ndarray]:
        '''
        Determine the trigger times, automatically convert from
            frame number to recording index by subtracting the
            firt frame number

        Note that not all recordings will have triggers, so
            may return None if doesn't exist

        :return:
        '''

        valid_keys = set(self.file_handle.keys())
        if HArrayDataReader.TRIGGER_KEY in valid_keys:
            trigger_times = self.file_handle[HArrayDataReader.TRIGGER_KEY][:]
            return trigger_times - self.first_frame_num
        return None

    def determine_electrical_gain(self) -> float:
        return self.file_handle[HArrayDataReader.SETTINGS_KEY][HArrayDataReader.GAIN_SUBKEY][0]

    def get_lsb(self) -> float:
        return self.file_handle[HArrayDataReader.SETTINGS_KEY][HArrayDataReader.LSB_SUBKEY][0]

    def get_cutoff_freq(self):
        return self.file_handle[HArrayDataReader.SETTINGS_KEY][HArrayDataReader.CUTOFF_FREQ_SUBKEY][0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file_handle.close()

    def close(self):
        self.file_handle.close()

