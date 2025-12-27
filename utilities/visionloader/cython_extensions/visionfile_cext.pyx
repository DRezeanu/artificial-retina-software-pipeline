def unpack_32bit_integers_from_bytearray (const unsigned char* input_buffer,
                                            Py_ssize_t num_integers_to_get,
                                            int[:] dest):

    cdef Py_ssize_t i
    cdef Py_ssize_t j = 0

    # need to flip the endianness
    cdef int temp
    for i in range(num_integers_to_get):
        temp = (input_buffer[j] << 24) | (input_buffer[j+1] << 16) | (input_buffer[j+2] << 8) | input_buffer[j+3]
        j += 4

        dest[i] = temp

    return



def unpack_alternating_32bit_float_from_bytearray (const unsigned char* input_buffer,
                                                    Py_ssize_t num_electrodes,
                                                    float[:,:] coord_dest):

    cdef Py_ssize_t i
    cdef Py_ssize_t j = 0

    # need to flip the endianness
    cdef int temp

    for i in range(num_electrodes):
        temp = (input_buffer[j] << 24) | (input_buffer[j+1] << 16) | (input_buffer[j+2] << 8) | input_buffer[j+3]
        j += 4

        coord_dest[i, 0] = (<float*> &temp)[0]

    for i in range(num_electrodes):
        temp = (input_buffer[j] << 24) | (input_buffer[j+1] << 16) | (input_buffer[j+2] << 8) | input_buffer[j+3]
        j += 4

        coord_dest[i, 1] = (<float*> &temp)[0]

    return


def unpack_ei_from_array (const unsigned char* input_buffer,
                        Py_ssize_t ei_num_samples,
                        Py_ssize_t ei_num_electrodes,
                        float[:,:] ei_output_buffer,
                        float[:,:] ei_error_buffer):

    cdef Py_ssize_t i, j
    cdef Py_ssize_t k = 0 

    cdef int ei_temp, error_temp

    cdef int a0, a1, a2, a3

    for i in range(ei_num_electrodes):
        for j in range(ei_num_samples):

            a0 = (input_buffer[k] & 0xFF) << 24
            a1 = (input_buffer[k+1] & 0xFF) << 16
            a2 = (input_buffer[k+2] & 0xFF) << 8
            a3 = input_buffer[k+3] & 0xFF

            ei_temp = a0 | a1 | a2 | a3
            k += 4

            a0 = (input_buffer[k] & 0xFF) << 24
            a1 = (input_buffer[k+1] & 0xFF) << 16
            a2 = (input_buffer[k+2] & 0xFF) << 8
            a3 = input_buffer[k+3] & 0xFF
            error_temp = a0 | a1 | a2 | a3
            k += 4

            ei_output_buffer[i,j] = (<float*> (&ei_temp))[0]
            ei_error_buffer[i,j] = (<float*> (&error_temp))[0]

    return


def unpack_64bit_float_from_bytearray (const unsigned char* input_buffer,
                                        Py_ssize_t num_doubles,
                                        Py_ssize_t start_offset,
                                        double[:] double_dest):

    cdef Py_ssize_t i
    cdef Py_ssize_t j = start_offset

    cdef long a0, a1, a2, a3

    cdef long temp

    for i in range(num_doubles):

        a0 = input_buffer[j] & 0xFF
        a1 = input_buffer[j+1] & 0xFF
        a2 = input_buffer[j+2] & 0xFF
        a3 = input_buffer[j+3] & 0xFF
        temp = (a0 << 56) | (a1 << 48) | (a2 << 40) | (a3 << 32)
        j += 4


        a0 = input_buffer[j] & 0xFF
        a1 = input_buffer[j+1] & 0xFF
        a2 = input_buffer[j+2] & 0xFF
        a3 = input_buffer[j+3] & 0xFF
        temp = temp | (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
        j += 4

        double_dest[i] = (<double*> &temp)[0]

    return


def pack_sta_from_bytearray (const unsigned char* input_buffer,
                            Py_ssize_t sta_width,
                            Py_ssize_t sta_height,
                            Py_ssize_t sta_depth,
                            float[:,:,:] red_value,
                            float[:,:,:] red_error,
                            float[:,:,:] green_value,
                            float[:,:,:] green_error,
                            float[:,:,:] blue_value,
                            float[:,:,:] blue_error):


    cdef Py_ssize_t idx = 12 # need to offset for one 64 bit double and one 32 bit integer
        # that we don't care about, corresponding to refresh time and STA depth, respectively

    cdef Py_ssize_t i, j, k # counters to iterate over the frames, width, and height, respectively

    cdef int temp # holds the float values temporarily before we cast and pack
    cdef int a0, a1, a2, a3 # holds the values taken from the byte array temporarily before we pack


    for i in range(sta_depth):

        idx += 16 # need to offset for two 32 bit integers and one 64 bit double,
            # corresponding to STA width, height, and stixel size, respectively
            # we already know these numbers from the header so we don't care

        for j in range(sta_width):
            for k in range(sta_height):

                a0 = input_buffer[idx] & 0xFF
                a1 = input_buffer[idx+1] & 0xFF
                a2 = input_buffer[idx+2] & 0xFF
                a3 = input_buffer[idx+3] & 0xFF
                temp = (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
                red_value[j, k, i] = (<float*> &temp)[0]
                idx += 4

                a0 = input_buffer[idx] & 0xFF
                a1 = input_buffer[idx+1] & 0xFF
                a2 = input_buffer[idx+2] & 0xFF
                a3 = input_buffer[idx+3] & 0xFF
                temp = (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
                red_error[j, k, i] = (<float*> &temp)[0]
                idx += 4

                a0 = input_buffer[idx] & 0xFF
                a1 = input_buffer[idx+1] & 0xFF
                a2 = input_buffer[idx+2] & 0xFF
                a3 = input_buffer[idx+3] & 0xFF
                temp = (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
                green_value[j, k, i] = (<float*> &temp)[0]
                idx += 4

                a0 = input_buffer[idx] & 0xFF
                a1 = input_buffer[idx+1] & 0xFF
                a2 = input_buffer[idx+2] & 0xFF
                a3 = input_buffer[idx+3] & 0xFF
                temp = (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
                green_error[j, k, i] = (<float*> &temp)[0]
                idx += 4

                a0 = input_buffer[idx] & 0xFF
                a1 = input_buffer[idx+1] & 0xFF
                a2 = input_buffer[idx+2] & 0xFF
                a3 = input_buffer[idx+3] & 0xFF
                temp = (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
                blue_value[j, k, i] = (<float*> &temp)[0]
                idx += 4

                a0 = input_buffer[idx] & 0xFF
                a1 = input_buffer[idx+1] & 0xFF
                a2 = input_buffer[idx+2] & 0xFF
                a3 = input_buffer[idx+3] & 0xFF
                temp = (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
                blue_error[j, k, i] = (<float*> &temp)[0]
                idx += 4


    return
