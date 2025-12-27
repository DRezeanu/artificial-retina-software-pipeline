import struct
import numpy as np
import os
import bin2py.cython_extensions.bin2py_cythonext as bin2py_cythonext
import warnings

from typing import Dict, Tuple, List, Optional, Any, Union, Set, BinaryIO

NBYTES_32BIT = 4  # type: int


# header binary tags
class HeaderTag:
    HEADER_LENGTH_TAG = 0  # type: int
    TIME_TAG = 1  # type: int
    COMMENT_TAG = 2  # type: int
    FORMAT_TAG = 3  # type: int
    ARRAY_ID_TAG = 4  # type: int
    FREQUENCY_TAG = 5  # type: int
    TRIGGER_TAG = 6  # type: int
    # deprecated according to vision java code comments

    DATASET_IDENTIFIER_TAG = 7  # type: int
    TRIGGER_TAG_V2 = 8  # type: int
    DATA_TAG = 499  # type: int
    FILE_TYPE = 0x512  # type: int


class HeaderSizeNBytes:
    HEADER_LENGTH_BYTES = 4  # type: int
    TIME_LENGTH_BYTES = 12  # type: int
    FORMAT_LENGTH_BYTES = 4  # type: int
    ARRAY_LENGTH_BYTES = 8  # type: int
    FREQUENCY_LENGTH_BYTES = 4  # type: int
    TRIGGER_LENGTH_BYTES = 8  # type: int
    TRIGGER_V2_LENGTH_BYTES = 16  # type: int
    DATA_TAG_LENGTH_BYTES = 4  # type: int


class FakeArrayID:
    '''
    based on the Matlab function fake_array_id
    '''

    BOARD_ID_512 = 504  # type: int
    BOARD_ID_519 = 1501  # type: int
    BOARD_ID_519_120UM = 1601  # type: int
    BOARD_ID_RECONFIGURABLE = 9999  # type: int
    # this is just convention now i guess


class ArrayInfo519:
    N_ELECTRODES = 520  # type: int
    N_BYTES_PER_SAMPLE = 780  # type: int
    SAMPLE_FREQ = 20000  # type: int


class ArrayInfo512:
    N_ELECTRODES = 513  # type: int
    N_BYTES_PER_SAMPLE = 770  # type: int
    SAMPLE_FREQ = 20000  # type: int


class BinDataEncoderDecoder:

    def __init__(self,
                 n_electrodes: int,
                 n_bytes_per_sample: int,
                 is_row_major: bool = False) -> None:

        self._N_BYTES_PER_SAMPLE = n_bytes_per_sample  # type: int
        self._N_ELECTRODES = n_electrodes  # type: int
        self._is_row_major = is_row_major

    @classmethod
    def construct_from_header(cls,
                              header: 'PyBinHeader',
                              is_row_major: bool = False) -> 'BinDataEncoderDecoder':

        '''
        Currently handles cases for the 30 um 519 board, and the 60 um 512 board only
        '''

        if header.array_id >= 1500 and header.array_id < 2500:
            # we have a 30 um 519 board
            assert (header.num_electrodes == ArrayInfo519.N_ELECTRODES), \
                'incorrect number of electrodes for 519 electrode 30 um board'
            return BinDataEncoderDecoder(ArrayInfo519.N_ELECTRODES,
                                         ArrayInfo519.N_BYTES_PER_SAMPLE,
                                         is_row_major=is_row_major)

        elif header.array_id >= 500 and header.array_id < 1500:
            # we have a 60 um 512 board
            assert (header.num_electrodes == ArrayInfo512.N_ELECTRODES), \
                'incorrect number of electrodes for 512 electrode 60 um board'
            return BinDataEncoderDecoder(ArrayInfo512.N_ELECTRODES,
                                         ArrayInfo512.N_BYTES_PER_SAMPLE,
                                         is_row_major=is_row_major)

        else:
            # we have an arbitrary number of electrodes
            n_bytes_per_sample = -1
            if header.num_electrodes % 2 == 0:
                n_bytes_per_sample = 3 * header.num_electrodes // 2
            else:
                n_bytes_per_sample = 2 + (header.num_electrodes - 1) * 3 // 2

            return BinDataEncoderDecoder(header.num_electrodes,
                                         n_bytes_per_sample,
                                         is_row_major=is_row_major)

    def parse_samples(self,
                      raw_samples_only_bytes_data: Union[bytes, bytearray],
                      num_samples: int,
                      data_output: np.ndarray,
                      row_offset: int) -> np.ndarray:
        '''
        Unpacks num_samples number of samples from raw sample data to a numpy array

        Since this is meant to interact with the current .bin format and vision, there are
            exactly two different ways to read samples in vision.
        Case 1: there are an even number of electrodes (including the TTL channel)
        Case 2: there are an odd number of electrodes (including the TTL channel)
        We just need to take care of these two cases, because that's how vision will deal with
            the data. As long as the header displays the correct number of electrodes,
            vision should be able to handle .bin files with arbitrary numbers of electrodes

        Args:
            raw_samples_only_bytes_data (bytes) : bytes array corresponding to the raw
                bin data that needs to be unpacked. The length of this bytes array should
                correspond exactly to num_samples (i.e. the first byte in this parameter must
                correspond to the first byte of the first sample to unpack, and the bytes array
                must be long enough to contain num_samples samples worth of data)

            num_samples (int) : number of samples to unpack

            data_output (numpy array) : expected shape (nrow >= row_offset + num_samples, n_electrodes)
                dtype='>H' which is a signed short (signed 16 bit integer)

                if self._is_row_major == True, expected_shape is (n_electrodes, ncol >= row_offset),
                dtype='>H', which is a signed short (signed 16 bit integer)

                Will write data into the array, changing its contents

            row_offset (int) : row of data_output that we should begin writing to. Method will write
                into rows (or columns, if is_row_major == True) with indices row_offset up
                to (row_offset + num_samples - 1)


        Returns: same numpy array as data_output, that has the extra data copied into it
            dtype='>H' which is a signed short (signed 16 bit integer)
        '''
        if self._N_ELECTRODES % 2 == 0:

            if not self._is_row_major:

                data_output = bin2py_cythonext.unpack_bin_even_num_electrodes(raw_samples_only_bytes_data,
                                                                              data_output,
                                                                              num_samples,
                                                                              row_offset)
            else:

                data_output = bin2py_cythonext.unpack_bin_even_num_electrodes_row_major(raw_samples_only_bytes_data,
                                                                                        data_output,
                                                                                        num_samples,
                                                                                        row_offset)

        else:

            if not self._is_row_major:
                data_output = bin2py_cythonext.unpack_bin_odd_num_electrodes(raw_samples_only_bytes_data,
                                                                             data_output,
                                                                             num_samples,
                                                                             row_offset)
            else:

                data_output = bin2py_cythonext.unpack_bin_odd_num_electrodes_row_major(raw_samples_only_bytes_data,
                                                                                       data_output,
                                                                                       num_samples,
                                                                                       row_offset)
        return data_output

    def parse_samples_for_electrode(self,
                                    raw_samples_only_bytes_data: Union[bytes, bytearray],
                                    electrode_index: int,
                                    num_samples: int,
                                    data_output: np.ndarray,
                                    row_offset: int) -> np.ndarray:

        if self._N_ELECTRODES % 2 == 0:
            data_output = bin2py_cythonext.unpack_single_electrode_even_num_electrodes(
                raw_samples_only_bytes_data,
                data_output,
                electrode_index,
                self._N_ELECTRODES,
                num_samples,
                row_offset)
        else:
            data_output = bin2py_cythonext.unpack_single_electrode_odd_num_electrodes(raw_samples_only_bytes_data,
                                                                                      data_output,
                                                                                      electrode_index,
                                                                                      self._N_ELECTRODES,
                                                                                      num_samples,
                                                                                      row_offset)

        return data_output

    def write_bin_data(self,
                       bin_file_obj: BinaryIO,
                       data_array: np.ndarray) -> None:
        '''
        Write the contents of data_array to the current end of bin file binary stream bin_file_obj

        Assumes that the header has already been written correctly.

        Args:
            bin_file_obj : open binary file stream corresponding to bin file. Must be writeable in append mode
            data_array (numpy array) : shape is (num_samples, num_electrodes). Includes the TTL channel. 
                Must have the correct shape or will throw an exception

        '''

        if self._N_ELECTRODES % 2 == 0:
            '''
            even number of electrodes, the 519 board will go through this case
            arbitrary number of electrodes with even number of electrodes will also go through this case
            '''
            if self._is_row_major:
                num_electrodes, num_samples = data_array.shape
                assert num_electrodes == self._N_ELECTRODES, \
                    "data for writing bin file (row-major) does not have correct shape"

                to_write = bytearray(self._N_BYTES_PER_SAMPLE * num_samples)
                to_write = bin2py_cythonext.pack_bin_sample_even_num_electrodes_row_major(data_array, to_write)
                bin_file_obj.write(to_write)

            else:
                # this is the column-major case
                num_samples, num_electrodes = data_array.shape
                assert num_electrodes == self._N_ELECTRODES, \
                    "data for writing bin file (column-major) does not have correct shape"

                to_write = bytearray(self._N_BYTES_PER_SAMPLE * num_samples)
                to_write = bin2py_cythonext.pack_bin_sample_even_num_electrodes(data_array, to_write)
                bin_file_obj.write(to_write)

        else:
            '''
            odd number of electrodes, the 512 board will go through this case
            arbitrary number of electrodes with odd number of electrodes will also go through this case
            '''

            if self._is_row_major:
                num_electrodes, num_samples = data_array.shape
                assert num_electrodes == self._N_ELECTRODES, \
                    "data for writing bin file (row-major) does not have correct shape"

                to_write = bytearray(self._N_BYTES_PER_SAMPLE * num_samples)
                to_write = bin2py_cythonext.pack_bin_sample_odd_num_electrodes_row_major(data_array, to_write)
                bin_file_obj.write(to_write)

            else:
                # column major case (the original codepath)
                num_samples, num_electrodes = data_array.shape
                assert num_electrodes == self._N_ELECTRODES, \
                    "data for writing bin file (column-major) does not have correct shape"

                to_write = bytearray(self._N_BYTES_PER_SAMPLE * num_samples)
                to_write = bin2py_cythonext.pack_bin_sample_odd_num_electrodes(data_array, to_write)
                bin_file_obj.write(to_write)


### define the header data ####_N_BYTES_PER_SAMPLE
class PyBinHeader:

    def __init__(self,
                 header_length: int = -1,
                 time_base: int = -1,
                 seconds_time: int = -1,
                 comment: Optional[str] = None,
                 dataset_identifier: Optional[str] = None,
                 dformat: int = -1,
                 array_id: int = -1,
                 num_electrodes: int = -1,
                 frequency: int = -1,
                 n_samples: int = -1,
                 trigger_contents: Optional[bytes] = None,
                 trigger_v2_contents: Optional[bytes] = None):

        self.header_length = header_length  # type: int
        self.time_base = time_base  # type: int
        self.seconds_time = seconds_time  # type: int

        if comment is None:
            self.comment = ''  # type: str
        else:
            self.comment = comment  # type: str

        if dataset_identifier is None:
            self.dataset_identifier = ''  # type: str
        else:
            self.dataset_identifier = dataset_identifier  # type: str

        self.format = dformat  # type: int
        self.array_id = array_id  # type: int
        self.num_electrodes = num_electrodes  # type: int
        self.frequency = frequency  # type: int
        self.n_samples = n_samples  # type: int

        self.trigger_contents = trigger_contents
        self.trigger_v2_contents = trigger_v2_contents

    @classmethod
    def make_header_from_parameters(cls,
                                    time_base: int,
                                    seconds_time: int,
                                    comment: str,
                                    dataset_identifier: str,
                                    dformat: int,
                                    array_id: int,
                                    num_electrodes: int,
                                    frequency: int,
                                    n_samples: int,
                                    trigger_tag: Optional[bytes] = None,
                                    trigger_tag_v2: Optional[bytes] = None) -> 'PyBinHeader':

        # the problem we need to solve here is that we need to calculate the length of the header
        if len(comment) % 2 != 0:
            comment = comment + ' '
        comment_n_bytes = len(comment)

        header_tag_front = 2 * NBYTES_32BIT

        header_length = 0
        header_length += header_tag_front + HeaderSizeNBytes.HEADER_LENGTH_BYTES  # header length tag
        header_length += header_tag_front + HeaderSizeNBytes.TIME_LENGTH_BYTES  # time tag
        header_length += header_tag_front + comment_n_bytes
        header_length += header_tag_front + HeaderSizeNBytes.FORMAT_LENGTH_BYTES
        header_length += header_tag_front + HeaderSizeNBytes.ARRAY_LENGTH_BYTES
        header_length += header_tag_front + HeaderSizeNBytes.FREQUENCY_LENGTH_BYTES
        header_length += header_tag_front + len(dataset_identifier)
        header_length += header_tag_front + HeaderSizeNBytes.DATA_TAG_LENGTH_BYTES

        # no longer skipping the trigger tags
        if trigger_tag is not None:
            assert len(trigger_tag) == HeaderSizeNBytes.TRIGGER_LENGTH_BYTES, "Trigger tag v1 length incorrect"
            header_length += HeaderSizeNBytes.TRIGGER_LENGTH_BYTES
        if trigger_tag_v2 is not None:
            assert len(trigger_tag_v2) == HeaderSizeNBytes.TRIGGER_V2_LENGTH_BYTES, \
                "Trigger tag v2 length incorrect"
            header_length += HeaderSizeNBytes.TRIGGER_V2_LENGTH_BYTES

        return cls(header_length=header_length,
                   time_base=time_base,
                   seconds_time=seconds_time,
                   comment=comment,
                   dataset_identifier=dataset_identifier,
                   dformat=dformat,
                   array_id=array_id,
                   num_electrodes=num_electrodes,
                   frequency=frequency,
                   n_samples=n_samples,
                   trigger_contents=trigger_tag,
                   trigger_v2_contents=trigger_tag_v2)

    @classmethod
    def make_519_header(cls,
                        time_base: int,
                        seconds_time: int,
                        comment: str,
                        dataset_identifier: str,
                        dformat: int,
                        n_samples: int) -> 'PyBinHeader':

        return cls.make_header_from_parameters(time_base,
                                               seconds_time,
                                               comment,
                                               dataset_identifier,
                                               dformat,
                                               FakeArrayID.BOARD_ID_519,
                                               ArrayInfo519.N_ELECTRODES,
                                               ArrayInfo519.SAMPLE_FREQ,
                                               n_samples)

    @classmethod
    def make_519_header_120um(cls,
                              time_base: int,
                              seconds_time: int,
                              comment: str,
                              dataset_identifier: str,
                              dformat: int,
                              n_samples) -> 'PyBinHeader':
        return cls.make_header_from_parameters(time_base,
                                               seconds_time,
                                               comment,
                                               dataset_identifier,
                                               dformat,
                                               FakeArrayID.BOARD_ID_519_120UM,
                                               ArrayInfo519.N_ELECTRODES,
                                               ArrayInfo519.SAMPLE_FREQ,
                                               n_samples)

    @classmethod
    def make_512_header(cls,
                        time_base: int,
                        seconds_time: int,
                        comment: str,
                        dataset_identifier: str,
                        dformat: int,
                        n_samples) -> 'PyBinHeader':

        return cls.make_header_from_parameters(time_base,
                                               seconds_time,
                                               comment,
                                               dataset_identifier,
                                               dformat,
                                               FakeArrayID.BOARD_ID_512,
                                               ArrayInfo512.N_ELECTRODES,
                                               ArrayInfo512.SAMPLE_FREQ,
                                               n_samples)

    def generate_header_in_binary(self) -> bytes:

        '''
        Generates header as byte array from the data contained in the object

        Returns:
            binary string corresponding to header

        '''

        header_as_list = []

        header_length_packed = struct.pack('>III',
                                           HeaderTag.HEADER_LENGTH_TAG,
                                           HeaderSizeNBytes.HEADER_LENGTH_BYTES,
                                           self.header_length)
        header_as_list.append(header_length_packed)

        time_tag_packed = struct.pack('>IIIQ', HeaderTag.TIME_TAG,
                                      HeaderSizeNBytes.TIME_LENGTH_BYTES,
                                      self.time_base,
                                      self.seconds_time)
        header_as_list.append(time_tag_packed)

        comment_tag_packed_header_only = struct.pack('>II',
                                                     HeaderTag.COMMENT_TAG,
                                                     len(self.comment))
        header_as_list.append(comment_tag_packed_header_only)
        header_as_list.append(self.comment.encode('utf-8'))

        format_tag_packed = struct.pack('>III',
                                        HeaderTag.FORMAT_TAG,
                                        HeaderSizeNBytes.FORMAT_LENGTH_BYTES,
                                        self.format)
        header_as_list.append(format_tag_packed)

        array_id_tag_packed = struct.pack('>IIII',
                                          HeaderTag.ARRAY_ID_TAG,
                                          HeaderSizeNBytes.ARRAY_LENGTH_BYTES,
                                          self.num_electrodes,
                                          self.array_id)
        header_as_list.append(array_id_tag_packed)

        frequency_tag_packed = struct.pack('>III',
                                           HeaderTag.FREQUENCY_TAG,
                                           HeaderSizeNBytes.FREQUENCY_LENGTH_BYTES,
                                           self.frequency)
        header_as_list.append(frequency_tag_packed)

        # updated to write the trigger tags if they are found and set
        if self.trigger_contents is not None:
            assert len(
                self.trigger_contents) == HeaderSizeNBytes.TRIGGER_LENGTH_BYTES, "Trigger tag v1 length incorrect"
            trigger_tag_packed = struct.pack('>II',
                                             HeaderTag.TRIGGER_TAG,
                                             HeaderSizeNBytes.TRIGGER_LENGTH_BYTES)
            header_as_list.append(trigger_tag_packed)
            header_as_list.append(self.trigger_contents)
        if self.trigger_v2_contents is not None:
            assert len(
                self.trigger_v2_contents) == HeaderSizeNBytes.TRIGGER_V2_LENGTH_BYTES, "Trigger tag v2 length incorrect"
            trigger_tag_v2_packed = struct.pack('>II',
                                                HeaderTag.TRIGGER_TAG_V2,
                                                HeaderSizeNBytes.TRIGGER_V2_LENGTH_BYTES)
            header_as_list.append(trigger_tag_v2_packed)
            header_as_list.append(self.trigger_v2_contents)

        dataset_iden_tag_packed_front = struct.pack('>II',
                                                    HeaderTag.DATASET_IDENTIFIER_TAG,
                                                    len(self.dataset_identifier))
        header_as_list.append(dataset_iden_tag_packed_front)
        header_as_list.append(self.dataset_identifier.encode('utf-8'))

        data_tag_packed = struct.pack('>III',
                                      HeaderTag.DATA_TAG,
                                      HeaderSizeNBytes.DATA_TAG_LENGTH_BYTES,
                                      self.n_samples)

        header_as_list.append(data_tag_packed)

        header_binary_string = b''.join(header_as_list)

        assert len(header_binary_string) == self.header_length, \
            "Generated binary header length {0} does not match object header length {1}".format(
                len(header_binary_string),
                self.header_length
            )

        return header_binary_string

    @classmethod
    def construct_from_bytearray(cls,
                                 bin_file_bytearray: Union[bytes, bytearray]) -> 'PyBinHeader':

        # assume that the header starts at the beginning of the bytearray
        curr_ind = 0
        num_bytes_bytearray = len(bin_file_bytearray)

        header_length = -1  # in bytes
        time_base = -1
        seconds_time = -1
        comment = None  # actual type is str
        dataset_identifier = None  # actual type is str
        dformat = -1
        num_electrodes = -1
        array_id = -1
        frequency = -1
        n_samples = -1

        trigger_contents = None  # don't know what actual type is, will be bytes (raw from orig file)
        trigger_v2_contents = None  # don't know what actual tyep is, will be bytes (raw from orig file)

        # parse the header tags #

        # parse the first header tag
        tag, size = struct.unpack('>II', bin_file_bytearray[curr_ind:curr_ind + NBYTES_32BIT * 2])
        curr_ind += NBYTES_32BIT * 2
        assert tag == HeaderTag.HEADER_LENGTH_TAG, 'No header tag'
        assert size == HeaderSizeNBytes.HEADER_LENGTH_BYTES, 'Header tag has incorrect size'

        # this tells us how long the header is, and hence where in the file the data section starts
        header_length, = struct.unpack('>I',
                                       bin_file_bytearray[curr_ind:curr_ind + HeaderSizeNBytes.HEADER_LENGTH_BYTES])
        curr_ind += HeaderSizeNBytes.HEADER_LENGTH_BYTES

        # parse all the other tags
        while tag != HeaderTag.DATA_TAG and curr_ind < num_bytes_bytearray:
            # DATA_TAG is the last tag, after we've handled that
            # we exit the loop. Also make sure we don't run out of bounds of the bytearray
            # since the bytearray might contain just the header only

            tag, size = struct.unpack('>II', bin_file_bytearray[curr_ind:curr_ind + NBYTES_32BIT * 2])
            curr_ind += NBYTES_32BIT * 2

            # vision doesn't assume anything about the order of the tags
            # after the header tag, so I won't either
            if tag == HeaderTag.TIME_TAG:
                assert size == HeaderSizeNBytes.TIME_LENGTH_BYTES, 'Time tag has incorrect size'
                time_base, seconds_time = struct.unpack('>IQ', bin_file_bytearray[curr_ind:curr_ind + NBYTES_32BIT * 3])
                curr_ind += NBYTES_32BIT * 3

            elif tag == HeaderTag.COMMENT_TAG:
                assert size % 2 == 0, "Comment header size is not even"
                comment = bin_file_bytearray[curr_ind:curr_ind + size].decode('utf-8')
                curr_ind += size

            elif tag == HeaderTag.FORMAT_TAG:
                assert size == HeaderSizeNBytes.FORMAT_LENGTH_BYTES, "Format tag has incorrect size"
                dformat, = struct.unpack('>I', bin_file_bytearray[curr_ind:curr_ind + NBYTES_32BIT])
                curr_ind += NBYTES_32BIT

            elif tag == HeaderTag.ARRAY_ID_TAG:
                assert size == HeaderSizeNBytes.ARRAY_LENGTH_BYTES, "Array id tag has incorrect size"
                num_electrodes, array_id = struct.unpack('>II',
                                                         bin_file_bytearray[curr_ind:curr_ind + NBYTES_32BIT * 2])
                curr_ind += 2 * NBYTES_32BIT

            elif tag == HeaderTag.FREQUENCY_TAG:
                assert size == HeaderSizeNBytes.FREQUENCY_LENGTH_BYTES, "Frequency tag has incorrect size"
                frequency, = struct.unpack('>I', bin_file_bytearray[curr_ind:curr_ind + NBYTES_32BIT])
                curr_ind += NBYTES_32BIT

            elif tag == HeaderTag.TRIGGER_TAG:
                if (size != HeaderSizeNBytes.TRIGGER_LENGTH_BYTES):
                    warnings.warn("Trigger v1 tag may be incorrect! Undefined behavior possible but unlikely.")
                # assert size == HeaderSizeNBytes.TRIGGER_LENGTH_BYTES, "Trigger v1 tag (deprecated) has incorrect size"
                # breaks for old datasets for some reason

                # I can't figure out how this field is being used
                # but we want to support writing through headers
                # that are loaded from real data, so we need to copy this quantity

                # FIXME hardcoded to be consistent with the next line
                # I cannot figure out what this field is or why it breaks for old data
                trigger_contents = bin_file_bytearray[curr_ind:curr_ind + (NBYTES_32BIT * 2)]

                curr_ind += NBYTES_32BIT * 2  # FIXME doesn't do anything with the contents of the tag.
                # Vision doesn't either so should be safe to ignore

            elif tag == HeaderTag.TRIGGER_TAG_V2:
                assert size == HeaderSizeNBytes.TRIGGER_V2_LENGTH_BYTES, "Trigger v2 tag has incorrect size"

                trigger_v2_contents = bin_file_bytearray[curr_ind:curr_ind + size]
                curr_ind += NBYTES_32BIT * 4  # FIXME Doesn't do anything with the contents of the tag
                # vision doesn't either so should be safe to ignore

            elif tag == HeaderTag.DATASET_IDENTIFIER_TAG:
                dataset_identifier = bin_file_bytearray[curr_ind:curr_ind + size].decode('utf-8')
                curr_ind += size

            elif tag == HeaderTag.DATA_TAG:
                assert size == HeaderSizeNBytes.DATA_TAG_LENGTH_BYTES, "data tag has incorrect size"
                n_samples, = struct.unpack('>I', bin_file_bytearray[curr_ind:curr_ind + NBYTES_32BIT])
                curr_ind += NBYTES_32BIT

            else:
                assert False, "Unknown tag {0}, don't know how to proceed".format(tag)

        return cls(header_length=header_length,
                   time_base=time_base,
                   seconds_time=seconds_time,
                   comment=comment,
                   dataset_identifier=dataset_identifier,
                   dformat=dformat,
                   array_id=array_id,
                   num_electrodes=num_electrodes,
                   frequency=frequency,
                   n_samples=n_samples,
                   trigger_contents=trigger_contents,
                   trigger_v2_contents=trigger_v2_contents)

    @classmethod
    def construct_from_binfile(cls,
                               bin_file_obj: BinaryIO) -> 'PyBinHeader':

        '''
        Loads and parses bin file header. Need to pass in binary file stream object created elsewhere

        Args:
            bin_file_obj (io.IOBase) : stream of bytes corresponding to the bin file.

        Returns:
            PyBinHeader, corresponding to the parsed header of the bin file

        '''

        # remember the starting position of the file pointer in the bin file before we start moving around
        initial_binfile_pos = bin_file_obj.tell()

        bin_file_obj.seek(0)  # go to the beginning of the bin file

        header_length = -1  # in bytes
        time_base = -1
        seconds_time = -1
        comment = None  # actual type is str
        dataset_identifier = None  # actual type is str
        dformat = -1
        num_electrodes = -1
        array_id = -1
        frequency = -1
        n_samples = -1

        trigger_contents = None
        trigger_v2_contents = None

        # parse the header tags #

        # parse the first header tag
        tag, size = struct.unpack('>II', bin_file_obj.read(NBYTES_32BIT * 2))  # uint32, big-endian
        assert tag == HeaderTag.HEADER_LENGTH_TAG, 'No header tag'
        assert size == HeaderSizeNBytes.HEADER_LENGTH_BYTES, 'Header tag has incorrect size'

        # this tells us how long the header is, and hence where in the file the data section starts
        header_length, = struct.unpack('>I', bin_file_obj.read(HeaderSizeNBytes.HEADER_LENGTH_BYTES))

        # parse all the other tags
        while tag != HeaderTag.DATA_TAG:  # DATA_TAG is the last tag, after we've handled that
            # we exit the loop

            tag, size = struct.unpack('>II', bin_file_obj.read(NBYTES_32BIT * 2))
            # size, = struct.unpack('>I', bin_file_obj.read(NBYTES_32BIT))

            # vision doesn't assume anything about the order of the tags
            # after the header tag, so I won't either
            if tag == HeaderTag.TIME_TAG:
                assert size == HeaderSizeNBytes.TIME_LENGTH_BYTES, 'Time tag has incorrect size'
                time_base, seconds_time = struct.unpack('>IQ', bin_file_obj.read(NBYTES_32BIT * 3))

            elif tag == HeaderTag.COMMENT_TAG:
                assert size % 2 == 0, "Comment header size is not even"
                comment = bin_file_obj.read(size).decode('utf-8')

            elif tag == HeaderTag.FORMAT_TAG:
                assert size == HeaderSizeNBytes.FORMAT_LENGTH_BYTES, "Format tag has incorrect size"
                dformat, = struct.unpack('>I', bin_file_obj.read(NBYTES_32BIT))

            elif tag == HeaderTag.ARRAY_ID_TAG:
                assert size == HeaderSizeNBytes.ARRAY_LENGTH_BYTES, "Array id tag has incorrect size"
                num_electrodes, = struct.unpack('>I', bin_file_obj.read(NBYTES_32BIT))
                array_id, = struct.unpack('>I', bin_file_obj.read(NBYTES_32BIT))

            elif tag == HeaderTag.FREQUENCY_TAG:
                assert size == HeaderSizeNBytes.FREQUENCY_LENGTH_BYTES, "Frequency tag has incorrect size"
                frequency, = struct.unpack('>I', bin_file_obj.read(NBYTES_32BIT))

            elif tag == HeaderTag.TRIGGER_TAG:
                if (size != HeaderSizeNBytes.TRIGGER_LENGTH_BYTES):
                    warnings.warn("Trigger v1 tag may be incorrect! Undefined behavior possible but unlikely.")
                # assert size == HeaderSizeNBytes.TRIGGER_LENGTH_BYTES, "Trigger v1 tag (deprecated) has incorrect size"
                # breaks for old datasets
                trigger_contents = bin_file_obj.read(
                    NBYTES_32BIT * 2)  # FIXME doesn't do anything with the contents of the tag.
                # Vision doesn't either so should be safe to ignore

            elif tag == HeaderTag.TRIGGER_TAG_V2:
                assert size == HeaderSizeNBytes.TRIGGER_V2_LENGTH_BYTES, "Trigger v2 tag has incorrect size"
                trigger_v2_contents = bin_file_obj.read(
                    NBYTES_32BIT * 4)  # FIXME Doesn't do anything with the contents of the tag
                # vision doesn't either so should be safe to ignore

            elif tag == HeaderTag.DATASET_IDENTIFIER_TAG:
                dataset_identifier = bin_file_obj.read(size).decode('utf-8')

            elif tag == HeaderTag.DATA_TAG:
                assert size == HeaderSizeNBytes.DATA_TAG_LENGTH_BYTES, "data tag has incorrect size"
                n_samples, = struct.unpack('>I', bin_file_obj.read(NBYTES_32BIT))

            else:
                print(tag)
                assert False, "Unknown tag, don't know how to proceed"

        bin_file_obj.seek(initial_binfile_pos)  # go back to where we were in the binfile

        return cls(header_length=header_length,
                   time_base=time_base,
                   seconds_time=seconds_time,
                   comment=comment,
                   dataset_identifier=dataset_identifier,
                   dformat=dformat,
                   array_id=array_id,
                   num_electrodes=num_electrodes,
                   frequency=frequency,
                   n_samples=n_samples,
                   trigger_contents=trigger_contents,
                   trigger_v2_contents=trigger_v2_contents)


class PyBinFileReader:
    '''
    Class for interacting with .bin files in Python.

    Supports parsing header, getting arbitrary data from bin file

    Note: handling of all the filestreams, including opening and closing them, needs to be handled in this class

  
    Usage:
    with PyBinFileReader('/Volumes/Data/...') as pbfr:
        pass


    '''

    def __init__(self,
                 bin_file_path: str,
                 bin_file_ext: str = '.bin',
                 chunk_samples: int = 10000,
                 is_row_major: bool = False):

        '''
        Instance attributes worth knowing about:

        self.bin_filestream_list (list of file handles) : list of file handles, one for each of the 
            bin files for the particular dataset. Should be in order (data00000 first, then data000001, ...)

        self.sample_number_list (list of int) : list of start/end sample indices for each file. Contains n+1
            items, given that we have n bin files. Contains the first sample index of each bin file, as well as
            the imaginary first sample of the imaginary (n+1)th bin file, which signifies the end.

        self.chunk_samples (int) : number of samples that should be read in a single chunk, to minimize
            total number of disk accesses

        '''

        self.chunk_samples = chunk_samples  # number of samples we should grab at a time from disk

        self.is_row_major = is_row_major  # type: bool

        self.bin_filestream_list = []  # type: List[BinaryIO]
        # list of binfiles corresponding to the dataset

        filesize_bytes_list = []
        # size of each of the .bin files in bytes, needed to compute the
        # number of samples in each .bin file

        # determine whether path is a .bin file, or a folder containing a bunch of .bin files
        if os.path.isdir(bin_file_path):

            bin_file_path_list = []

            # it is a directory, we need to get all of the .bin files within the directory
            for filename in os.listdir(bin_file_path):

                fpath = os.path.join(bin_file_path, filename)
                if not os.path.isdir(fpath) and os.path.splitext(fpath)[1] == bin_file_ext:
                    bin_file_path_list.append(fpath)

            # sort in increasing order (data00000.bin goes first, etc.)
            bin_file_path_list.sort()  # should be a string sort

            # get file pointers to each of the bin files, store in the list in order
            for fpath in bin_file_path_list:
                filesize_bytes_list.append(os.stat(fpath).st_size)
                self.bin_filestream_list.append(open(fpath, 'rb'))

            if len(bin_file_path_list) == 0:
                raise IOError('No bin files found in folder {0}'.format(bin_file_path))

        else:
            # it is not a directory, so by default assume that it is a .bin file and open it
            filesize_bytes_list.append(os.stat(bin_file_path).st_size)
            self.bin_filestream_list.append(open(bin_file_path, 'rb'))

        self.header = PyBinHeader.construct_from_binfile(
            self.bin_filestream_list[0])  # the header is in the first bin file

        # figure out which decoder is needed based on the header
        self.decoder = BinDataEncoderDecoder.construct_from_header(self.header, self.is_row_major)

        self.sample_number_list = [0, ]  # stores the sample number at which each bin file starts
        # the last number in this list corresponds to one sample past the last sample

        # now figure out which samples are in which file
        scount = 0
        for i, fsize in enumerate(filesize_bytes_list):

            nbytes_file_body = fsize
            if i == 0:  # first file, need to account for header size
                nbytes_file_body = fsize - self.header.header_length

            n_samples_in_file = nbytes_file_body // self.decoder._N_BYTES_PER_SAMPLE
            scount += n_samples_in_file
            self.sample_number_list.append(scount)

        '''
        assert self.sample_number_list[-1] <= self.header.n_samples, \
            "Number of samples {0} in the dataset should be less than or equal to the number {1} stated in the header".format(
                self.sample_number_list[-1], self.header.n_samples)
        '''

        # it is possible that the recording gets interuppted, and we have fewer samples in the actual dataset
        # than the header claims there are

        # assert self.sample_number_list[-1] == self.header.n_samples, \
        #        "Number of samples {0} in the dataset doesn't match the number {1} given in the header".format(self.sample_number_list[-1], self.header.n_samples)

    def _get_file_index_for_sample(self,
                                   sample_num: int) -> int:

        if sample_num < 0 or sample_num >= self.sample_number_list[-1]:
            raise IndexError('Sample number is out of bounds')

        findex = 0
        while (findex + 1) < len(self.sample_number_list) and \
                self.sample_number_list[findex + 1] < sample_num:
            findex += 1
        return findex

    def _count_remaining_samples(self,
                                 fnum: int,
                                 sample_num: int) -> int:
        '''
        Count the number of unread samples in file corresponding to fnum
            after sample_num, including sample_num

        Args:
            fnum (int) : index of file number
            sample_num (int) : sample number (0 corresponds to the first sample
                    of the first file)

        Returns:
            int, number of samples remaining in the file. Could be negative
        '''
        return self.sample_number_list[fnum + 1] - sample_num

    def _get_sample_offset(self,
                           fnum: int,
                           sample_num: int) -> int:
        '''
        Given the sample number w.r.t. the entire dataset, calculate the sample number
            relative to the specific file

        Args:
            fnum (int) : index of file number
            sample_num : sample number (0 corresponds to the first sample of the first
                  file)

        Returns:
            int, sample offset number
        
        '''
        return sample_num - self.sample_number_list[fnum]

    def _get_start_sample_for_file(self,
                                   fnum: int) -> int:
        return self.sample_number_list[fnum]

    def _get_last_sample_for_file(self,
                                  fnum: int) -> int:
        return self.sample_number_list[fnum + 1] - 1

    def get_data_for_electrode(self,
                               electrode_num: int,
                               start_sample_num: int,
                               num_samples: int) -> np.ndarray:

        '''
        Gets num_samples number of samples, starting at sample number start_sample_num, for electrode electrode_num
        Returns numpy array with dimensions (num_samples, ), 
                dtype='>H' which is a signed short (signed 16 bit)

        Currently only supports 512 electrode and 519 electrode boards


        Args:
            electrode_num (int) : electrode number to get data for, 0 is TTL
            start_sample_num (int) : sample number to start at
            num_samples (int) : number of samples to get data for

        Returns:
            numpy array, with shape (num_samples, n_electrodes) corresponding to the decoded data
                datatype of the array is np.int16, which is a signed 16 bit integer
        '''

        # calculate number of the last sample
        end_sample_num = start_sample_num + num_samples - 1

        # make sure that the dataset has enough samples
        assert end_sample_num < self.sample_number_list[-1], 'Exceeds the length of the dataset'

        # allocate memory for output
        output_array = np.zeros((num_samples,), dtype=np.int16)

        # figure out which bin file to start in
        start_file_no = self._get_file_index_for_sample(start_sample_num)

        # begin unpacking, starting from the first file 
        fnum = start_file_no
        freading = self.bin_filestream_list[fnum]

        # go to the first sample but don't read anything
        sample_offset = self._get_sample_offset(fnum, start_sample_num)
        if fnum == 0:  # first file, we need to account for the header length
            freading.seek(self.header.header_length + \
                          sample_offset * self.decoder._N_BYTES_PER_SAMPLE)
        else:
            freading.seek(sample_offset * self.decoder._N_BYTES_PER_SAMPLE)

        j = 0
        while j < num_samples:

            # if we are already at the end of a file, advance the file pointer
            # this is guaranteed to not go out of bounds because we already checked that
            # we are not getting more samples than there are in the dataset
            if self._count_remaining_samples(fnum, j + start_sample_num) <= 0:
                fnum += 1
                freading = self.bin_filestream_list[fnum]

                # obviously this cannot be the first file, therefore just start at beginning
                freading.seek(0)

            # we read the min number of samples of the following:
            # 1. Remaining number of samples needed for the method call (ns_total_left_to_read)
            # 2. Remaining number of unread samples in this particular .bin file (ns_reamining_in_file)
            # 3. The chunk size (self.chunk_samples)
            ns_remaining_in_file = self._count_remaining_samples(fnum, j + start_sample_num)
            ns_total_left_to_read = num_samples - j

            ns_to_read = min(self.chunk_samples, ns_remaining_in_file, ns_total_left_to_read)
            # read the raw data and parse, copy into output_array
            samples_raw = freading.read(self.decoder._N_BYTES_PER_SAMPLE * ns_to_read)
            self.decoder.parse_samples_for_electrode(samples_raw, electrode_num, ns_to_read, output_array, j)

            j += ns_to_read

        return output_array

    def get_data(self,
                 start_sample_num: int,
                 num_samples: int) -> np.ndarray:

        '''
        Gets num_samples number of samples, starting at sample number start_sample_num, for all electrodes
        Returns numpy array with dimensions (num_samples, n_electrodes), 
                dtype='>H' which is a signed short (signed 16 bit)

        Currently only supports 512 electrode and 519 electrode boards


        Args:
            start_sample_num (int) : sample number to start at
            num_samples (int) : number of samples to get data for

        Returns:
            numpy array, with shape (num_samples, n_electrodes) corresponding to the decoded data
                datatype of the array is np.int16, which is a signed 16 bit integer
        '''

        # calculate number of the last sample
        end_sample_num = start_sample_num + num_samples - 1

        # make sure that the dataset has enough samples
        assert end_sample_num < self.sample_number_list[-1], 'Exceeds the length of the dataset'

        # allocate memory for output
        output_array = None
        if self.is_row_major:
            output_array = np.zeros((self.decoder._N_ELECTRODES, num_samples),
                                    dtype=np.int16)
        else:
            output_array = np.zeros((num_samples, self.decoder._N_ELECTRODES),
                                    dtype=np.int16)

        # figure out which bin file to start in
        start_file_no = self._get_file_index_for_sample(start_sample_num)

        # begin unpacking, starting from the first file 
        fnum = start_file_no
        freading = self.bin_filestream_list[fnum]

        # go to the first sample but don't read anything
        sample_offset = self._get_sample_offset(fnum, start_sample_num)
        if fnum == 0:  # first file, we need to account for the header length
            freading.seek(self.header.header_length + \
                          sample_offset * self.decoder._N_BYTES_PER_SAMPLE)
        else:
            freading.seek(sample_offset * self.decoder._N_BYTES_PER_SAMPLE)

        j = 0
        while j < num_samples:

            # if we are already at the end of a file, advance the file pointer
            # this is guaranteed to not go out of bounds because we already checked that
            # we are not getting more samples than there are in the dataset
            if self._count_remaining_samples(fnum, j + start_sample_num) <= 0:
                fnum += 1
                freading = self.bin_filestream_list[fnum]

                # obviously this cannot be the first file, therefore just start at beginning
                freading.seek(0)

            # we read the min number of samples of the following:
            # 1. Remaining number of samples needed for the method call (ns_total_left_to_read)
            # 2. Remaining number of unread samples in this particular .bin file (ns_reamining_in_file)
            # 3. The chunk size (self.chunk_samples)
            ns_remaining_in_file = self._count_remaining_samples(fnum, j + start_sample_num)
            ns_total_left_to_read = num_samples - j

            ns_to_read = min(self.chunk_samples, ns_remaining_in_file, ns_total_left_to_read)

            # read the raw data and parse, copy into output_array
            samples_raw = freading.read(self.decoder._N_BYTES_PER_SAMPLE * ns_to_read)
            self.decoder.parse_samples(samples_raw, ns_to_read, output_array, j)

            j += ns_to_read

        return output_array

    @property
    def length(self) -> int:
        return self.sample_number_list[-1]

    @property
    def num_electrodes(self) -> int:
        return (self.header.num_electrodes - 1)

    @property
    def array_id(self) -> int:
        return (self.header.array_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        for fp in self.bin_filestream_list:
            fp.close()


class PyBinFileWriter:

    def __init__(self,
                 header: PyBinHeader,
                 path_to_write: str,
                 dataset_name: str,
                 bin_ext: str = '.bin',
                 bin_file_n_samples: int = 2400000,
                 is_row_major: bool = False) -> None:

        '''
        Construct a PyBinFileWriter object given a valid header

        Args:
            header (PyBinHeader) : valid header object for the bin file
            path_to_write (str) : path to the location where dataset should be written. Must either be an existing
                            folder or not exist at all (cannot be an existing file)
            dataset_name (str) : name of the dataset / dataset folder. Note that path_to_write/dataset_name cannot
                            exist before the constructor is called
            bin_ext (str) : extension of the files that we want to write
            bin_file_n_samples (int) : maximum number of samples allowed to be in a single .bin file
            data_format_row_major (bool) : whether the np.ndarray matrix of data we're writing out is
                            currently in row-major order (each channel is a row
                            of the output matrix where adjacent samples are next to each other in memory,
                            so performing operations like filtering is much much faster). Default False

        Example usage:
        with PyBinFileWrite(header, path, dsname) as pbfw:
            pbfw.write_samples(data)

        '''

        self.header = header  # type: PyBinHeader
        self.bin_file_n_samples = bin_file_n_samples  # type: int

        self.encoder = BinDataEncoderDecoder.construct_from_header(header,
                                                                   is_row_major=is_row_major)  # type: BinDataEncoderDecoder

        self.path_to_write = path_to_write  # type: str
        self.dataset_name = dataset_name  # type: str

        self.bin_ext = bin_ext  # type: str
        self.bin_file_n_samples = bin_file_n_samples  # type: int

        self.dataset_folder_path = os.path.join(self.path_to_write, self.dataset_name)

        self.active_file_number = 0  # type: int
        self.actively_writing_file = None  # type: BinaryIO # type: ignore
        self.n_samples_written_to_active_file = 0  # type: int

        self.is_row_major = is_row_major  # type: bool

    def write_samples(self, data_array: np.ndarray) -> None:

        '''
        Writes samples to bin file / folder
        '''

        n_samples_to_write, n_electrodes_to_write = None, None
        if self.is_row_major:
            n_electrodes_to_write, n_samples_to_write = data_array.shape
        else:
            n_samples_to_write, n_electrodes_to_write = data_array.shape

        assert n_electrodes_to_write == self.encoder._N_ELECTRODES, \
            "cannot write different number of electrodes {0} than specified by the header {1}".format(
                n_electrodes_to_write, self.encoder._N_ELECTRODES)

        # do some special handling if this is the first time we're writing to the very first file
        if self.active_file_number == 0 and self.n_samples_written_to_active_file == 0:

            # find the directory path_to_write
            # if it doesn't exist, try to create it
            # if it does exist and is a file, raise an exception
            # after finding or creating the directory, create another folder corresponding to dataset_name if it doesn't exist
            # if it does exist, raise an Exception
            # then create a bin file corresponding to the first bin file of the dataset and write the header
            if os.path.exists(self.path_to_write) and not os.path.isdir(self.path_to_write):
                raise Exception("Cannot write to {0}, would overwrite existing file".format(self.path_to_write))
            elif not os.path.exists(self.path_to_write):
                os.mkdir(self.path_to_write)
            if os.path.exists(self.dataset_folder_path):
                raise Exception("Cannot create dataset folder {0} in {1}, would overwrite existing file/folder".format(
                    self.dataset_name, self.path_to_write))

            os.mkdir(self.dataset_folder_path)

            self.actively_writing_file = open(os.path.join(self.dataset_folder_path,
                                                           "{0}{1:03d}{2}".format(self.dataset_name,
                                                                                  self.active_file_number,
                                                                                  self.bin_ext)), 'wb')

            self.n_samples_written_to_active_file = 0
            self.actively_writing_file.write(self.header.generate_header_in_binary())

        # we may need to break up the data over multiple files

        i = 0
        while i < n_samples_to_write:
            # check if we need to create and write to a new bin file and handle if necessary
            if self.n_samples_written_to_active_file >= self.bin_file_n_samples:
                # first close the old bin file
                self.actively_writing_file.close()

                # then create the new bin file
                self.active_file_number += 1
                self.actively_writing_file = open(os.path.join(self.dataset_folder_path,
                                                               "{0}{1:03d}{2}".format(self.dataset_name,
                                                                                      self.active_file_number,
                                                                                      self.bin_ext)), 'wb')

                # then update the count for number of samples we've written to the new file
                # note that we don't need to write a header to files that are not the first
                self.n_samples_written_to_active_file = 0

            # figure out how many samples we can write
            max_number_allowed_to_write = min(n_samples_to_write - i,
                                              self.bin_file_n_samples - self.n_samples_written_to_active_file)

            if self.is_row_major:
                self.encoder.write_bin_data(self.actively_writing_file,
                                            data_array[:, i:i + max_number_allowed_to_write])
            else:
                self.encoder.write_bin_data(self.actively_writing_file,
                                            data_array[i:i + max_number_allowed_to_write, :])

            i += max_number_allowed_to_write
            self.n_samples_written_to_active_file += max_number_allowed_to_write

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.actively_writing_file is not None:
            self.actively_writing_file.close()

    def close(self):
        if self.actively_writing_file is not None:
            self.actively_writing_file.close()
