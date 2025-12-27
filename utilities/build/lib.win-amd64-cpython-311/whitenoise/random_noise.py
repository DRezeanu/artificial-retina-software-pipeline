import numpy as np
import scipy.stats as stats

from . import noise_frame_generator
import xml.etree.ElementTree as ET

from typing import List, Dict, Union, Tuple


class RNTypeEnum:
    RGB_BINARY = 1
    BW_BINARY = 2

    RGB_GAUSSIAN = 3
    BW_GAUSSIAN = 2


class RandomNoiseFrameGenerator:
    '''
    Class to generate random noise frames in the exact same way that Photons generates a random frame

    Pretty much a direct port of the class Random_Noise from the Photons source

    @author Eric Wu

    '''

    def __init__(self,
                 seed_init: int,
                 stixel_width: int,
                 stixel_height: int,
                 field_width: int,  # width in stixels
                 field_height: int,  # height in stixels
                 probability: float,  # probability of showing a pixel
                 background_rgb: np.ndarray,
                 rgb_weights: np.ndarray,
                 independent_colors: bool,
                 is_binary: bool,
                 jitter: bool,
                 refresh_interval: int,
                 map_file_name: Union[str, None] = None):

        '''

        :param seed_init: (int) random seed value, i.e. 11111 or 22222
        :param stixel_width: (int) width of a single stixel
        :param stixel_height: (int) height of a single stixel
        :param field_width: (int) width of stimulus in stixels
        :param field_height: (int) height of stimulus in stixels
        :param probability: (float) probability of showing a pixel
        :param background_rgb:
        :param rgb_weights:
        :param independent_colors: (bool) color or BW, True if color
        :param is_binary: (bool) whether or not stimulus is binary
        :param jitter: (bool) whether or not the stimulus is jittered
        :param refresh_interval:
        :param map_file_name:
        '''

        self.init_seed = seed_init
        self.rng = noise_frame_generator.JavaRandSequence(seed_init)

        self.stixel_width = stixel_width
        self.stixel_height = stixel_height

        self.field_width = field_width
        self.field_height = field_height

        self.refresh_interval = refresh_interval

        self.probability = probability

        self.background_rgb = background_rgb  # shape (3, )
        self.rgb_weights = rgb_weights  # shape (3, )

        self.independent_colors = independent_colors
        self.is_binary = is_binary
        self.jitter = jitter

        self.jitter_rng = None
        if self.jitter:
            self.jitter_rng = noise_frame_generator.JavaRandSequence(seed_init)

        self.map = None

        self.noise_type = None  # type: Union[None, int]
        self.n_bits = None  # type: Union[None, int]
        tmp = None  # type: Union[None, np.ndarray]
        # initialize lookup tables, etc. based on parameters
        if self.is_binary:  # binary stimulus:
            if self.independent_colors:  # color stimulus

                self.noise_type = RNTypeEnum.RGB_BINARY
                self.n_bits = 3
                tmp = np.array([[1, 1, 1],
                                [1, 1, -1],
                                [1, -1, 1],
                                [1, -1, -1],
                                [-1, 1, 1],
                                [-1, 1, -1],
                                [-1, -1, 1],
                                [-1, -1, -1]])

                tmp = tmp * self.rgb_weights[None, :] + self.background_rgb[None, :]

            else:  # BW stimulus

                self.noise_type = RNTypeEnum.BW_BINARY
                self.n_bits = 1
                tmp = np.array([[1, 1, 1, ],
                                [-1, -1, -1]])

                tmp = tmp * self.rgb_weights[None, :] + self.background_rgb[None, :]

        else:  # Gaussian stimulus, currently untested code path
            if self.independent_colors:
                self.noise_type = RNTypeEnum.RGB_GAUSSIAN
            else:
                self.noise_type = RNTypeEnum.BW_GAUSSIAN

            self.n_bits = 3
            tmp = stats.norm.ppf(np.r_[1:256] / 257, loc=0, scale=1)

            tmp = np.tile(tmp, (1, 3)) * self.rgb_weights[None, :] + self.background_rgb[None, :]

        self.lut = np.rint(tmp * 255).astype(np.uint8)
        self.map_back_rgb = np.rint(255 * self.background_rgb).astype(np.int8)

        # do something about the map file stuff
        self.map = np.empty((0, 0), dtype=np.uint16)  # type: Union[None, np.ndarray]
        self.m_width, self.m_height = None, None
        if map_file_name is not None:
            pass  # FIXME later
        else:
            self.m_width = self.field_width
            self.m_height = self.field_height

        self.span_height = self.field_height * self.stixel_height
        self.span_width = self.field_width * self.stixel_width

    @classmethod
    def construct_from_xml(cls, xml_path, is_jitter=False) -> 'RandomNoiseFrameGenerator':

        tree = ET.parse(xml_path)
        root = tree.getroot()

        width = -1  # type: int
        height = -1  # type: int
        seed = -1  # type: int
        color_type = -1  # type: int
        noise_type = -1  # type: int
        random_number_generator = -1  # type: int
        contrast_sigma = -1.0  # type: float

        probability = 1.0  # type: float

        stixel_size_x = -1  # type: int
        stixel_size_y = -1  # type: int
        refresh_interval = -1  # type: int

        for param_group in root:

            if param_group.attrib['name'] == 'Make White Noise Movie':

                for parameter in param_group:
                    param_name = parameter.attrib['name']
                    param_value = parameter.attrib['value']

                    if param_name == 'Width':
                        width = int(param_value)
                    elif param_name == 'Height':
                        height = int(param_value)
                    elif param_name == 'Seed':
                        seed = int(param_value)
                    elif param_name == 'ColorType':
                        color_type = int(float(param_value))
                    elif param_name == 'NoiseType':
                        noise_type = int(float(param_value))
                    elif param_name == 'RandomNumberGenerator':
                        random_number_generator = int(float(param_value))
                    elif param_name == 'ContrastSigma':
                        contrast_sigma = float(param_value)
                    elif param_name == 'Probability':
                        probability = float(param_value)

            elif param_group.attrib['name'] == 'Calculate Auxiliary Parameters':
                # this entire section isn't strictly necessary
                # we should still be able to generate frames
                # even if this data isn't there or is wrong
                for nested_param_group in param_group:

                    if nested_param_group.attrib['name'] == 'Set Movie':

                        for param in nested_param_group:
                            param_name = param.attrib['name']
                            param_value = param.attrib['value']

                            if param_name == 'pixelsPerStixelX':
                                stixel_size_x = int(param_value)
                            elif param_name == 'pixelsPerStixelY':
                                stixel_size_y = int(param_value)
                            elif param_name == 'refreshInterval':
                                refresh_interval = int(param_value)

        # now check that the important values are valid
        assert width != -1, 'Invalid width in movie xml'
        assert height != -1, 'Invalid height in movie xml'
        assert seed != -1, 'Invalid seed in movie xml'
        assert (color_type == 0 or color_type == 1), 'Invalid color type in movie xml. Separated not implemented.'
        assert noise_type != -1, 'Invalid noise type in movie xml'
        assert random_number_generator == 2, 'Invalid random number generator in movie xml'
        # note that we only implemented rng Java V2, which is what Photons uses
        assert contrast_sigma != -1.0, 'Invalid contrast sigma in movie xml'

        independent_colors = (color_type == 0)
        is_binary = (noise_type == 0)

        return RandomNoiseFrameGenerator(seed,
                                         stixel_size_x,
                                         stixel_size_y,
                                         width,
                                         height,
                                         probability,
                                         0.5 * np.array([1.0, 1.0, 1.0]),
                                         contrast_sigma * np.array([1.0, 1.0, 1.0]),
                                         independent_colors,
                                         is_binary,
                                         is_jitter,
                                         refresh_interval,
                                         None)

    def reset_seed_to_beginning(self):
        self.rng = noise_frame_generator.JavaRandSequence(self.init_seed)

    def advance_seed_n_frames(self, n_frames: int) -> None:
        noise_frame_generator.advance_seed_n_frames(self.rng,
                                                    self.m_width,
                                                    self.m_height,
                                                    self.noise_type,
                                                    self.n_bits,
                                                    self.probability,
                                                    n_frames)
        return None

    def generate_next_frame(self) -> np.ndarray:

        '''
        Generate the next frame, and advance the seed accordingly

        This generates the same matrix as what Photons does when it generates
            a single frame to display. The differences are that the array
            dimension order is different to accomodate numpy/C array order rather
            than MATLAB/Fortran order, and that we ignore the luminance channel

        Includes RGB and luminance values
        :return: np.ndarray of uint8, shape (height, width, 3)
        '''

        if self.jitter:
            if self.jitter_rng is None:
                assert False, "Did not initialize jitter RNG"
            return noise_frame_generator.draw_upsampled_jittered_frame(self.rng,
                                                                       self.jitter_rng,
                                                                       self.field_width,
                                                                       self.field_height,
                                                                       self.lut,
                                                                       self.map,
                                                                       self.background_rgb,
                                                                       self.m_width,
                                                                       self.m_height,
                                                                       self.noise_type,
                                                                       self.n_bits,
                                                                       self.probability,
                                                                       self.stixel_width,
                                                                       self.stixel_height
                                                                       )

        return noise_frame_generator.draw_random_single_frame(self.rng,
                                                              self.field_width,
                                                              self.field_height,
                                                              self.lut,
                                                              self.map,
                                                              self.background_rgb,
                                                              self.m_width,
                                                              self.m_height,
                                                              self.noise_type,
                                                              self.n_bits,
                                                              self.probability)

    @property
    def output_dims(self):

        if self.jitter:
            return (self.span_height, self.span_width)
        else:
            return (self.field_height, self.field_width)

    def generate_block_of_frames(self, n_frames) -> np.ndarray:

        '''
        Generate a sequence of frames, and advance the seed accordingly

        This generates the sequence of frames (should match the output of Photons,
            if we removed the luminance channel and stacked consecutive frames on top
            of each other
        :param n_frames: int, number of frames to generate
        :return: np.ndarray of uint8, shape (n_frames, height, width, 3)
        '''
        if self.jitter:
            if self.jitter_rng is None:
                assert False, "Did not initialize jitter RNG"
            return noise_frame_generator.draw_upsampled_jittered_consecutive_frames(self.rng,
                                                                                    self.jitter_rng,
                                                                                    self.field_width,
                                                                                    self.field_height,
                                                                                    n_frames,
                                                                                    self.lut,
                                                                                    self.map,
                                                                                    self.background_rgb,
                                                                                    self.m_width,
                                                                                    self.m_height,
                                                                                    self.noise_type,
                                                                                    self.n_bits,
                                                                                    self.probability,
                                                                                    self.stixel_width,
                                                                                    self.stixel_height
                                                                                    )

        return noise_frame_generator.draw_consecutive_frames(self.rng,
                                                             self.field_width,
                                                             self.field_height,
                                                             n_frames,
                                                             self.lut,
                                                             self.map,
                                                             self.background_rgb,
                                                             self.m_width,
                                                             self.m_height,
                                                             self.noise_type,
                                                             self.n_bits,
                                                             self.probability)
