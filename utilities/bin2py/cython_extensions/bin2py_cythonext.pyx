cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pack_bin_sample_even_num_electrodes (short[:,:] data_array, unsigned char[:] output_buffer):
    
    cdef Py_ssize_t num_samples = data_array.shape[0]
    cdef Py_ssize_t num_electrodes = data_array.shape[1]

    cdef Py_ssize_t i, j

    cdef Py_ssize_t k = 0

    cdef short first, second
    cdef char b1, b2, b3

    for i in range(num_samples):
        for j in range(0, num_electrodes, 2):

            first = data_array[i,j] + 2048
            second = data_array[i,j+1] + 2048

            b1 = (first >> 4)
            b2 = ((first & 0xF) << 4) | (second >> 8)
            b3 = (second & 0xFF)

            output_buffer[k] = b1
            output_buffer[k+1] = b2
            output_buffer[k+2] = b3

            k += 3

    return output_buffer

@cython.boundscheck(False)
@cython.wraparound(False)
def pack_bin_sample_even_num_electrodes_row_major (short[:,:] data_array, unsigned char [:] output_buffer):

    cdef Py_ssize_t num_samples = data_array.shape[1]
    cdef Py_ssize_t num_electrodes = data_array.shape[0]

    cdef Py_ssize_t i, j

    cdef Py_ssize_t k = 0

    cdef short first, second
    cdef char b1, b2, b3

    for i in range(num_samples):
        for j in range(0, num_electrodes, 2):

            first = data_array[j,i] + 2048
            second = data_array[j+1,i] + 2048

            b1 = (first >> 4)
            b2 = ((first & 0xF) << 4) | (second >> 8)
            b3 = (second & 0xFF)

            output_buffer[k] = b1
            output_buffer[k+1] = b2
            output_buffer[k+2] = b3

            k += 3

    return output_buffer

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pack_bin_sample_odd_num_electrodes (short[:,:] data_array, unsigned char[:] output_buffer):

    cdef Py_ssize_t num_samples = data_array.shape[0]
    cdef Py_ssize_t num_electrodes = data_array.shape[1]

    cdef Py_ssize_t i, j

    cdef Py_ssize_t k = 0

    cdef short first, second
    cdef char b1, b2, b3


    for i in range(num_samples):

        # write the TTL channel
        b1 = data_array[i, 0] >> 8
        b2 = data_array[i, 0] & 0xFF

        output_buffer[k] = b1
        output_buffer[k+1] = b2

        k += 2

        # write each of the remaining channels
        for j in range(1, num_electrodes, 2):

            first = data_array[i, j] + 2048
            second = data_array[i, j+1] + 2048

            b1 = first >> 4
            b2 = (((first & 0xF) << 4) | (second >> 8))
            b3 = second & 0xFF

            output_buffer[k] = b1
            output_buffer[k+1] = b2
            output_buffer[k+2] = b3

            k += 3

    return output_buffer


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def pack_bin_sample_odd_num_electrodes_row_major (short[:,:] data_array, unsigned char[:] output_buffer):

    cdef Py_ssize_t num_samples = data_array.shape[1]
    cdef Py_ssize_t num_electrodes = data_array.shape[0]

    cdef Py_ssize_t i, j

    cdef Py_ssize_t k = 0

    cdef short first, second
    cdef char b1, b2, b3

    for i in range(num_samples):

        # write the TTL channel
        b1 = data_array[0,i] >> 8
        b2 = data_array[0,i] & 0xFF

        output_buffer[k] = b1
        output_buffer[k+1] = b2

        k += 2

        # write each of the remaining channels
        for j in range(1, num_electrodes, 2):

            first = data_array[j,i] + 2048
            second = data_array[j+1,i] + 2048

            b1 = first >> 4
            b2 = (((first & 0xF) << 4) | (second >> 8))
            b3 = second & 0xFF

            output_buffer[k] = b1
            output_buffer[k+1] = b2
            output_buffer[k+2] = b3

            k += 3

    return output_buffer

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def unpack_single_electrode_even_num_electrodes (const unsigned char* input_buffer,
                                                short[:] data_output,
                                                Py_ssize_t electrode_index,
                                                Py_ssize_t n_electrodes,
                                                Py_ssize_t num_samples,
                                                Py_ssize_t write_offset):


    cdef Py_ssize_t one_sample_nbytes = (3 * n_electrodes) >> 1

    cdef short b1, b2, b3

    cdef Py_ssize_t i
    cdef Py_ssize_t k = ((electrode_index & (~ 0x1)) * 3) >> 1

    if (electrode_index & 0x1) == 0x0:
        # even sample number, we want the first sample
        for i in range(num_samples):
            # get the correct two bytes from the buffer
            b1, b2 = input_buffer[k], input_buffer[k+1]
            k += one_sample_nbytes
            data_output[write_offset+i] = (((b1 << 4) | (b2 >> 4)) - 2048)
    else:
        for i in range(num_samples):
            b2, b3 = input_buffer[k+1], input_buffer[k+2]
            k += one_sample_nbytes
            data_output[write_offset+i] = ((((b2 & 0xF) << 8) | b3) - 2048)

    return data_output


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def unpack_single_electrode_odd_num_electrodes (const unsigned char* input_buffer,
                                                short[:] data_output,
                                                Py_ssize_t electrode_index,
                                                Py_ssize_t n_electrodes,
                                                Py_ssize_t num_samples,
                                                Py_ssize_t write_offset):

    cdef Py_ssize_t num_rec_electrodes = n_electrodes - 1
    cdef Py_ssize_t one_sample_nbytes = ((3 * num_rec_electrodes) >> 1) + 2

    cdef short b1, b2, b3
    cdef Py_ssize_t i

    # figure out which index we want to start at
    cdef Py_ssize_t k
    if electrode_index == 0:
        k = 0
        for i in range(num_samples):
            # just extract the TTL channel
            b1, b2 = input_buffer[k], input_buffer[k+1]
            k += one_sample_nbytes
            data_output[write_offset + i] = (b1 << 8) | (b2 & 0xFF)
    else:
        k = 2 + ((((electrode_index - 1) & (~0x1)) * 3) >> 1)

        # note that this is a little different from the above case
        # the odd number sample is now the first of the two samples
        # this is because we treat the TTL channel as a special thing
        
        if (electrode_index & 0x1) == 0x1:
            # odd sample number, we want the first sample
            for i in range(num_samples):
                # get the correct two bytes from the buffer
                b1, b2 = input_buffer[k], input_buffer[k+1]
                k += one_sample_nbytes
                data_output[write_offset+i] = (((b1 << 4) | (b2 >> 4)) - 2048)
        else:
            for i in range(num_samples):
                b2, b3 = input_buffer[k+1], input_buffer[k+2]
                k += one_sample_nbytes
                data_output[write_offset+i] = ((((b2 & 0xF) << 8) | b3) - 2048)


    return data_output


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def unpack_bin_even_num_electrodes (const unsigned char* input_buffer, 
                                    short[:,:] data_output, 
                                    Py_ssize_t num_samples,
                                    Py_ssize_t row_offset):

    '''
    Given an input buffer of bytes from the raw .bin file, unpack the samples into
        short array data_output. Assumes that the data is encoded in the format for
        an even number of electrodes (electrode count includes TTL channel)

    Args:
    
    input_buffer : input buffer of bytes, taken directly from the .bin file.
        The first item in input_buffer must be the first byte of the sample that we
        are interested in reading. Input_buffer must contain enough data that we can
        read at least num_samples worth of data. Python passes in a bytes object into
        this function
    data_output : array that the samples will be written into. Think of this as a C
        pointer to an array. Must be large enough to write num_samples worth of data
        into. Python passes in a numpy array of type int16 into this parameter. The
        number of columns of this array must match the number of electrodes (including
        the TTL channel)
    num_samples : number of samples to be read
    row_offset : how many rows into data_output we should start writing the first sample
        to. This enables us to allocate a single large array for data_output, and write
        separately to parts of it.
    


    even number of electrodes, the 519 board will go through this case
    arbitrary number of electrodes with even number of electrodes will also go through this case


    following unpacking algorithm courtesy of Nora:
    This is different in 519 array data. Channel 0 is no longer treated specially.
    Add 2048 to all the channels. Convert channel 0-1, 2-3, 4-5 etc. into three bytes each.

    To unpack, convert bytes 0-2, 3-5, 6-8 etc. into two samples each.
    Subtract 2048 from all the channels.
    Continue converting each set of 780 bytes into 520 electrodes until done.
    '''

    cdef Py_ssize_t i, j
    cdef Py_ssize_t k = 0

    cdef Py_ssize_t n_electrodes = data_output.shape[1]

    cdef short b1, b2, b3

    for i in range(num_samples):
        for j in range(0, n_electrodes, 2):
            # get three bytes from the buffer
            b1, b2, b3 = input_buffer[k], input_buffer[k+1], input_buffer[k+2]
            k += 3

            data_output[i+row_offset,j] = (((b1 << 4) | (b2 >> 4)) - 2048)
            data_output[i+row_offset,j+1] = ((((b2 & 0xF) << 8) | b3) - 2048)

    return data_output


def unpack_bin_odd_num_electrodes (const unsigned char* input_buffer, 
                                    short[:,:] data_output, 
                                    Py_ssize_t num_samples,
                                    Py_ssize_t row_offset):
    '''
    Given an input buffer of bytes from the raw .bin file, unpack the samples into
        short array data_output. Assumes that the data is encoded in the format for
        an odd number of electrodes (electrode count includes TTL channel)

    Args:
    
    input_buffer : input buffer of bytes, taken directly from the .bin file.
        The first item in input_buffer must be the first byte of the sample that we
        are interested in reading. Input_buffer must contain enough data that we can
        read at least num_samples worth of data. Python passes in a bytes object into
        this function
    data_output : array that the samples will be written into. Think of this as a C
        pointer to an array. Must be large enough to write num_samples worth of data
        into. Python passes in a numpy array of type int16 into this parameter. The
        number of columns of this array must match the number of electrodes (including
        the TTL channel)
    num_samples : number of samples to be read
    row_offset : how many rows into data_output we should start writing the first sample
        to. This enables us to allocate a single large array for data_output, and write
        separately to parts of it.

    odd number of electrodes, the 512 board will go through this case
    arbitrary number of electrodes with odd number of electrodes will also go through this case
    
    following this unpacking algorithm courtesy of Nora:
    Copy bytes 0,1 directly into the TTL signal (channel 0)
    Convert bytes 2-4, 5-7, 8-10 etc. into two samples each:
    s1=(b1<<4) + (b2 >>4);
    s2=(b2 & 0x000F) << 8 + b3;

    Subtract 2048 from all channels except zero.
    Continue converting each set of 770 bytes into 513 electrodes until done.
    '''

    cdef Py_ssize_t i, j
    cdef Py_ssize_t k = 0

    cdef Py_ssize_t n_electrodes = data_output.shape[1]

    cdef short b1, b2, b3

    for i in range(num_samples):

        # extract the TTL channel
        b1, b2 = input_buffer[k], input_buffer[k+1]
        k += 2

        data_output[i+row_offset,0] = (b1 << 8) | (b2 & 0xFF)

        for j in range(1, n_electrodes, 2):
            # get three bytes from the buffer
            b1, b2, b3 = input_buffer[k], input_buffer[k+1], input_buffer[k+2]
            k += 3

            data_output[i+row_offset,j] = (((b1 << 4) | (b2 >> 4)) - 2048)
            data_output[i+row_offset,j+1] = ((((b2 & 0xF) << 8) | b3) - 2048)

    return data_output



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def unpack_bin_even_num_electrodes_row_major (const unsigned char* input_buffer,
                                    short[:,:] data_output,
                                    Py_ssize_t num_samples,
                                    Py_ssize_t row_offset):

    '''
    Given an input buffer of bytes from the raw .bin file, unpack the samples into
        short array data_output. Assumes that the data is encoded in the format for
        an even number of electrodes (electrode count includes TTL channel)

    Args:

    input_buffer : input buffer of bytes, taken directly from the .bin file.
        The first item in input_buffer must be the first byte of the sample that we
        are interested in reading. Input_buffer must contain enough data that we can
        read at least num_samples worth of data. Python passes in a bytes object into
        this function
    data_output : array that the samples will be written into. Think of this as a C
        pointer to an array. Must be large enough to write num_samples worth of data
        into. Python passes in a numpy array of type int16 into this parameter. The
        number of columns of this array must match the number of electrodes (including
        the TTL channel)
    num_samples : number of samples to be read
    row_offset : how many rows into data_output we should start writing the first sample
        to. This enables us to allocate a single large array for data_output, and write
        separately to parts of it.



    even number of electrodes, the 519 board will go through this case
    arbitrary number of electrodes with even number of electrodes will also go through this case


    following unpacking algorithm courtesy of Nora:
    This is different in 519 array data. Channel 0 is no longer treated specially.
    Add 2048 to all the channels. Convert channel 0-1, 2-3, 4-5 etc. into three bytes each.

    To unpack, convert bytes 0-2, 3-5, 6-8 etc. into two samples each.
    Subtract 2048 from all the channels.
    Continue converting each set of 780 bytes into 520 electrodes until done.
    '''

    cdef Py_ssize_t i, j
    cdef Py_ssize_t k = 0

    cdef Py_ssize_t n_electrodes = data_output.shape[0]

    cdef short b1, b2, b3

    for i in range(num_samples):
        for j in range(0, n_electrodes, 2):
            # get three bytes from the buffer
            b1, b2, b3 = input_buffer[k], input_buffer[k+1], input_buffer[k+2]
            k += 3

            data_output[j,i+row_offset] = (((b1 << 4) | (b2 >> 4)) - 2048)
            data_output[j+1,i+row_offset] = ((((b2 & 0xF) << 8) | b3) - 2048)

    return data_output




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def unpack_bin_odd_num_electrodes_row_major (const unsigned char* input_buffer,
                                    short[:,:] data_output,
                                    Py_ssize_t num_samples,
                                    Py_ssize_t row_offset):
    '''
    Given an input buffer of bytes from the raw .bin file, unpack the samples into
        short array data_output. Assumes that the data is encoded in the format for
        an odd number of electrodes (electrode count includes TTL channel)

    Args:

    input_buffer : input buffer of bytes, taken directly from the .bin file.
        The first item in input_buffer must be the first byte of the sample that we
        are interested in reading. Input_buffer must contain enough data that we can
        read at least num_samples worth of data. Python passes in a bytes object into
        this function
    data_output : array that the samples will be written into. Think of this as a C
        pointer to an array. Must be large enough to write num_samples worth of data
        into. Python passes in a numpy array of type int16 into this parameter. The
        number of columns of this array must match the number of electrodes (including
        the TTL channel)
    num_samples : number of samples to be read
    row_offset : how many rows into data_output we should start writing the first sample
        to. This enables us to allocate a single large array for data_output, and write
        separately to parts of it.

    odd number of electrodes, the 512 board will go through this case
    arbitrary number of electrodes with odd number of electrodes will also go through this case

    following this unpacking algorithm courtesy of Nora:
    Copy bytes 0,1 directly into the TTL signal (channel 0)
    Convert bytes 2-4, 5-7, 8-10 etc. into two samples each:
    s1=(b1<<4) + (b2 >>4);
    s2=(b2 & 0x000F) << 8 + b3;

    Subtract 2048 from all channels except zero.
    Continue converting each set of 770 bytes into 513 electrodes until done.
    '''

    cdef Py_ssize_t i, j
    cdef Py_ssize_t k = 0

    cdef Py_ssize_t n_electrodes = data_output.shape[0]

    cdef short b1, b2, b3

    for i in range(num_samples):

        # extract the TTL channel
        b1, b2 = input_buffer[k], input_buffer[k+1]
        k += 2

        data_output[0,i+row_offset] = (b1 << 8) | (b2 & 0xFF)

        for j in range(1, n_electrodes, 2):
            # get three bytes from the buffer
            b1, b2, b3 = input_buffer[k], input_buffer[k+1], input_buffer[k+2]
            k += 3

            data_output[j,i+row_offset] = (((b1 << 4) | (b2 >> 4)) - 2048)
            data_output[j+1,i+row_offset] = ((((b2 & 0xF) << 8) | b3) - 2048)

    return data_output
