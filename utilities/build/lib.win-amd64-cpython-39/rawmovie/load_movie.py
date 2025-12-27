import numpy as np
import re

from typing import List, Tuple, Union, Dict, Optional
from . import rawmovie_ops

RGB_CONVERSION = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def _validate_crop(height: int, width: int,
                   crop_h_low: int, crop_h_high: int,
                   crop_w_low: int, crop_w_high: int) -> None:
    if crop_h_low < 0 or crop_h_low >= height:
        raise ValueError('crop_h_low={0} must be interval [{1}, {2})'.format(
            crop_h_low, 0, height))
    if crop_h_high < 0 or crop_h_high >= height:
        raise ValueError('crop_h_high={0} must be interval [{1}, {2})'.format(
            crop_h_high, 0, height))

    if (height - (crop_h_low + crop_h_high)) <= 0:
        raise ValueError('Image height is {0}, tried to crop {1} from height'.format(
            height, (crop_h_low + crop_h_high)))

    if crop_w_low < 0 or crop_w_low >= width:
        raise ValueError('crop_w_low={0} must be interval [{1}, {2})'.format(
            crop_w_low, 0, width))
    if crop_w_high < 0 or crop_w_high >= width:
        raise ValueError('crop_w_high={0} must be interval [{1}, {2})'.format(
            crop_w_high, 0, width))

    if (width - (crop_w_low + crop_w_high)) <= 0:
        raise ValueError('Image height is {0}, tried to crop {1} from height'.format(
            width, (crop_w_low + crop_w_high)))


class RawMovieReader:
    HEADER_READCHARS = 8000
    N_MOVIE_CHANNELS = 3

    def __init__(self,
                 rawmovie_filepath: str,
                 tab_char: int = 0x09,
                 new1_char: int = 0x0d,
                 new2_char: int = 0x0a,
                 chunk_n_frames: int = 2500,
                 crop_h_low: int = 0,
                 crop_h_high: int = 0,
                 crop_w_low: int = 0,
                 crop_w_high: int = 0):

        '''
        First parse the header, get the header size and all

        For that we want to load the front of the file as a
            string, not binary

        Then close the file, and reopen as binary

        :param rawmovie_filepath:
        '''

        header_size_regex = re.compile(r"header-size\s+(?P<size>\d+)")
        width_regex = re.compile(r"width\s+(?P<width>\d+)")
        height_regex = re.compile(r"height\s+(?P<height>\d+)")
        frames_generated_regex = re.compile(r"frames-generated\s+(?P<frames_generated>\d+)")

        self.header_length = -1  # type: int
        self.width = -1  # type: int
        self.height = -1  # type: int
        self.nframes = -1  # type: int

        self.chunk_n_frames = chunk_n_frames

        self.crop_h_low = crop_h_low
        self.crop_h_high = crop_h_high
        self.crop_w_low = crop_w_low
        self.crop_w_high = crop_w_high

        with open(rawmovie_filepath, 'r', encoding='latin1') as temp_rawmovie:

            # load first 8000 characters like in the MATLAB code
            front_piece = temp_rawmovie.read(RawMovieReader.HEADER_READCHARS)

            header_size_matches = re.search(header_size_regex, front_piece)
            if header_size_matches:
                self.header_length = int(header_size_matches.group('size'))
            else:
                assert False, "Could not get header length from {0}".format(rawmovie_filepath)

            width_matches = re.search(width_regex, front_piece)
            if width_matches:
                self.width = int(width_matches.group('width'))
            else:
                assert False, "Could not get frame width from {0}".format(rawmovie_filepath)

            height_matches = re.search(height_regex, front_piece)
            if height_matches:
                self.height = int(height_matches.group('height'))
            else:
                assert False, "Could not get frame height from {0}".format(rawmovie_filepath)

            nframes_matches = re.search(frames_generated_regex, front_piece)
            if nframes_matches:
                self.nframes = int(nframes_matches.group('frames_generated'))
            else:
                assert False, "Could not get frame count from {0}".format(rawmovie_filepath)

        # now load the raw movie file pointer
        self.rawmovie_fp = open(rawmovie_filepath, 'rb')

        # make sure that the crops are reasonable
        # and that we don't end up with negative dimensions
        _validate_crop(self.height, self.width,
                       self.crop_h_low, self.crop_h_high,
                       self.crop_w_low, self.crop_w_high)

        # compute crop dimensions
        self.low_h, self.high_h = self.crop_h_low, self.height - self.crop_h_high
        self.low_w, self.high_w = self.crop_w_low, self.width - self.crop_w_high

        # compute the output dimensions
        self.output_h = self.high_h - self.low_h
        self.output_w = self.high_w - self.low_w

        # compute size of single frame on disk
        self.single_frame_size = self.width * self.height * RawMovieReader.N_MOVIE_CHANNELS

    def get_all_frames(self) -> Tuple[np.ndarray, int]:
        return self.get_frame_sequence(0, self.nframes)

    def get_all_frames_bw(self) -> Tuple[np.ndarray, int]:
        return self.get_frame_sequence_bw(0, self.nframes)

    def get_frame_sequence(self,
                           start_frame: int,
                           end_frame: int) -> Tuple[np.ndarray, int]:

        end_frame = min(end_frame, self.nframes)

        # jump to the beginning of the data section
        output_matrix = np.zeros(
            (end_frame - start_frame, self.output_h, self.output_w, RawMovieReader.N_MOVIE_CHANNELS),
            dtype=np.uint8)

        write_offset_frames = 0
        self.rawmovie_fp.seek(self.header_length + self.single_frame_size * start_frame)
        for read_offset_frame in range(start_frame, end_frame, self.chunk_n_frames):
            n_frames_to_read = min(self.chunk_n_frames, end_frame - read_offset_frame)
            write_end = write_offset_frames + n_frames_to_read

            data_for_frames = self.rawmovie_fp.read(n_frames_to_read * self.single_frame_size)
            as_1d_np_array = np.frombuffer(data_for_frames, dtype=np.uint8)
            reshaped_frames = np.reshape(as_1d_np_array,
                                         (n_frames_to_read, self.height, self.width, RawMovieReader.N_MOVIE_CHANNELS))

            output_matrix[write_offset_frames:write_end, ...] = reshaped_frames[:, self.low_h:self.high_h,
                                                                self.low_w:self.high_w, :]
            write_offset_frames += n_frames_to_read

        return output_matrix, end_frame - start_frame

    def get_frame_sequence_bw(self,
                              start_frame: int,
                              end_frame: int) -> Tuple[np.ndarray, int]:

        end_frame = min(end_frame, self.nframes)

        # jump to the beginning of the data section
        output_matrix = np.zeros((end_frame - start_frame, self.output_h, self.output_w),
                                 dtype=np.float32)

        write_offset_frames = 0
        self.rawmovie_fp.seek(self.header_length + self.single_frame_size * start_frame)
        for read_offset_frame in range(start_frame, end_frame, self.chunk_n_frames):
            n_frames_to_read = min(self.chunk_n_frames, end_frame - read_offset_frame)
            write_end = write_offset_frames + n_frames_to_read

            data_for_frames = self.rawmovie_fp.read(n_frames_to_read * self.single_frame_size)
            as_1d_np_array = np.frombuffer(data_for_frames, dtype=np.uint8)
            reshaped_frames = np.reshape(as_1d_np_array,
                                         (n_frames_to_read, self.height, self.width, RawMovieReader.N_MOVIE_CHANNELS))

            output_matrix[write_offset_frames:write_end, ...] = rawmovie_ops.convert_color_to_bw_float32(
                reshaped_frames[:, self.low_h:self.high_h, self.low_w:self.high_w])

            write_offset_frames = write_end

        return output_matrix, end_frame - start_frame

    def _get_raw_frame_debug_mode(self, frameno: int) -> np.ndarray:
        # jump to the beginning of the data section

        frame_chunksize = self.width * self.height * RawMovieReader.N_MOVIE_CHANNELS

        self.rawmovie_fp.seek(self.header_length + frame_chunksize * frameno)

        data_for_frame = self.rawmovie_fp.read(frame_chunksize)
        as_1d_np_array = np.frombuffer(data_for_frame, dtype=np.uint8)

        return as_1d_np_array

    def get_single_frame(self, frameno: int) -> np.ndarray:

        frame_chunksize = self.width * self.height * RawMovieReader.N_MOVIE_CHANNELS

        self.rawmovie_fp.seek(self.header_length + frame_chunksize * frameno)

        data_for_frame = self.rawmovie_fp.read(frame_chunksize)
        as_1d_np_array = np.frombuffer(data_for_frame, dtype=np.uint8)
        reshaped_frame = np.reshape(as_1d_np_array, (self.height, self.width, RawMovieReader.N_MOVIE_CHANNELS))

        return reshaped_frame[self.low_h:self.high_h, self.low_w:self.high_w, :]

    def get_single_frame_bw(self, frameno: int) -> np.ndarray:
        # shape (height, width, 3) @ (1, 3, 1) -> (height, width, 1) -> (height, width)
        return (self.get_single_frame(frameno) @ RGB_CONVERSION).squeeze(2)[self.low_h:self.high_h,
               self.low_w:self.high_w]

    @property
    def num_frames(self):
        return self.nframes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.rawmovie_fp.close()

    def close(self):
        self.rawmovie_fp.close()


class RawMovieReader2:
    HEADER_READCHARS = 8000
    N_MOVIE_CHANNELS = 3

    def __init__(self,
                 rawmovie_filepath: str,
                 tab_char: int = 0x09,
                 new1_char: int = 0x0d,
                 new2_char: int = 0x0a,
                 chunk_n_frames: int = 2500):

        '''
        First parse the header, get the header size and all

        For that we want to load the front of the file as a
            string, not binary

        Then close the file, and reopen as binary

        :param rawmovie_filepath:
        '''

        header_size_regex = re.compile(r"header-size\s+(?P<size>\d+)")
        width_regex = re.compile(r"width\s+(?P<width>\d+)")
        height_regex = re.compile(r"height\s+(?P<height>\d+)")
        frames_generated_regex = re.compile(r"frames-generated\s+(?P<frames_generated>\d+)")

        self.header_length = -1  # type: int
        self.width = -1  # type: int
        self.height = -1  # type: int
        self.nframes = -1  # type: int

        self.chunk_n_frames = chunk_n_frames

        with open(rawmovie_filepath, 'r', encoding='latin1') as temp_rawmovie:

            # load first 8000 characters like in the MATLAB code
            front_piece = temp_rawmovie.read(RawMovieReader.HEADER_READCHARS)

            header_size_matches = re.search(header_size_regex, front_piece)
            if header_size_matches:
                self.header_length = int(header_size_matches.group('size'))
            else:
                assert False, "Could not get header length from {0}".format(rawmovie_filepath)

            width_matches = re.search(width_regex, front_piece)
            if width_matches:
                self.width = int(width_matches.group('width'))
            else:
                assert False, "Could not get frame width from {0}".format(rawmovie_filepath)

            height_matches = re.search(height_regex, front_piece)
            if height_matches:
                self.height = int(height_matches.group('height'))
            else:
                assert False, "Could not get frame height from {0}".format(rawmovie_filepath)

            nframes_matches = re.search(frames_generated_regex, front_piece)
            if nframes_matches:
                self.nframes = int(nframes_matches.group('frames_generated'))
            else:
                assert False, "Could not get frame count from {0}".format(rawmovie_filepath)

        # now load the raw movie file pointer
        self.rawmovie_fp = open(rawmovie_filepath, 'rb')

        # compute size of single frame on disk
        self.single_frame_size = self.width * self.height * RawMovieReader.N_MOVIE_CHANNELS

    def get_all_frames(self,
                       h_low_high: Optional[Tuple[int, int]] = None,
                       w_low_high: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int]:
        return self.get_frame_sequence(0, self.nframes,
                                       h_low_high=h_low_high,
                                       w_low_high=w_low_high)

    def get_all_frames_bw(self,
                          h_low_high: Optional[Tuple[int, int]] = None,
                          w_low_high: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int]:
        return self.get_frame_sequence_bw(0, self.nframes,
                                          h_low_high=h_low_high,
                                          w_low_high=w_low_high)

    def get_frame_sequence(self,
                           start_frame: int,
                           end_frame: int,
                           h_low_high: Optional[Tuple[int, int]] = None,
                           w_low_high: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int]:

        h_low_slice, h_high_slice = 0, self.height
        if h_low_high is not None:
            h_low_slice, h_high_slice = h_low_high

        w_low_slice, w_high_slice = 0, self.width
        if w_low_high is not None:
            w_low_slice, w_high_slice = w_low_high

        output_h = h_high_slice - h_low_slice
        output_w = w_high_slice - w_low_slice

        end_frame = min(end_frame, self.nframes)

        # jump to the beginning of the data section
        output_matrix = np.zeros(
            (end_frame - start_frame, output_h, output_w, RawMovieReader.N_MOVIE_CHANNELS),
            dtype=np.uint8)

        write_offset_frames = 0
        self.rawmovie_fp.seek(self.header_length + self.single_frame_size * start_frame)
        for read_offset_frame in range(start_frame, end_frame, self.chunk_n_frames):
            n_frames_to_read = min(self.chunk_n_frames, end_frame - read_offset_frame)
            write_end = write_offset_frames + n_frames_to_read

            data_for_frames = self.rawmovie_fp.read(n_frames_to_read * self.single_frame_size)
            as_1d_np_array = np.frombuffer(data_for_frames, dtype=np.uint8)
            reshaped_frames = np.reshape(as_1d_np_array,
                                         (n_frames_to_read, self.height, self.width, RawMovieReader.N_MOVIE_CHANNELS))

            output_matrix[write_offset_frames:write_end, ...] = reshaped_frames[:, h_low_slice:h_high_slice,
                                                                w_low_slice:w_high_slice, :]
            write_offset_frames += n_frames_to_read

        return output_matrix, end_frame - start_frame

    def get_frame_sequence_bw(self,
                              start_frame: int,
                              end_frame: int,
                              h_low_high: Optional[Tuple[int, int]] = None,
                              w_low_high: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int]:

        h_low_slice, h_high_slice = 0, self.height
        if h_low_high is not None:
            h_low_slice, h_high_slice = h_low_high

        w_low_slice, w_high_slice = 0, self.width
        if w_low_high is not None:
            w_low_slice, w_high_slice = w_low_high

        output_h = h_high_slice - h_low_slice
        output_w = w_high_slice - w_low_slice

        end_frame = min(end_frame, self.nframes)

        # jump to the beginning of the data section
        output_matrix = np.zeros((end_frame - start_frame, output_h, output_w), dtype=np.uint8)

        write_offset_frames = 0
        self.rawmovie_fp.seek(self.header_length + self.single_frame_size * start_frame)
        for read_offset_frame in range(start_frame, end_frame, self.chunk_n_frames):
            n_frames_to_read = min(self.chunk_n_frames, end_frame - read_offset_frame)
            write_end = write_offset_frames + n_frames_to_read

            data_for_frames = self.rawmovie_fp.read(n_frames_to_read * self.single_frame_size)
            as_1d_np_array = np.frombuffer(data_for_frames, dtype=np.uint8)
            reshaped_frames = np.reshape(as_1d_np_array,
                                         (n_frames_to_read, self.height, self.width, RawMovieReader.N_MOVIE_CHANNELS))

            before_integer = rawmovie_ops.convert_color_8bit_to_bw_float32_noncontig(
                reshaped_frames[:, h_low_slice:h_high_slice, w_low_slice:w_high_slice, :]).astype(np.uint8)

            output_matrix[write_offset_frames:write_end, ...] = before_integer
            write_offset_frames += n_frames_to_read

        return output_matrix, end_frame - start_frame

    @property
    def num_frames(self):
        return self.nframes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.rawmovie_fp.close()

    def close(self):
        self.rawmovie_fp.close()
