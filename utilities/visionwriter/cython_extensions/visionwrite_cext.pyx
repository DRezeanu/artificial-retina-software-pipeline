def pack_32bit_integers_to_bytearray (int[:] source,
                                      unsigned char[:] output_buffer):

    cdef Py_ssize_t i, j
    cdef unsigned char a, b, c, d

    cdef int temp

    j = 0
    for i in range(source.shape[0]):

        temp = source[i]

        a = (temp >> 24) & 0xFF
        b = (temp >> 16) & 0xFF
        c = (temp >> 8) & 0xFF
        d = temp & 0xFF

        output_buffer[j] = a
        output_buffer[j+1] = b
        output_buffer[j+2] = c
        output_buffer[j+3] = d

        j += 4

    return output_buffer


def pack_electrode_coordinates_globals (float[:,:] elec_coordinates,
                                        unsigned char[:] output_buffer):


    cdef Py_ssize_t n_electrodes = elec_coordinates.shape[0]
    cdef Py_ssize_t i, j

    cdef float temp
    cdef int temp_as_int

    cdef unsigned char a, b, c, d

    j = 0

    for i in range(n_electrodes):
        temp = elec_coordinates[i,0]
        temp_as_int = (<int*> &temp)[0]

        a = (temp_as_int >> 24) & 0xFF
        b = (temp_as_int >> 16) & 0xFF
        c = (temp_as_int >> 8) & 0xFF
        d = temp_as_int & 0xFF

        output_buffer[j] = a
        output_buffer[j+1] = b
        output_buffer[j+2] = c
        output_buffer[j+3] = d

        j += 4

    for i in range(n_electrodes):
        temp = elec_coordinates[i,1]
        temp_as_int = (<int*> &temp)[0]

        a = (temp_as_int >> 24) & 0xFF
        b = (temp_as_int >> 16) & 0xFF
        c = (temp_as_int >> 8) & 0xFF
        d = temp_as_int & 0xFF

        output_buffer[j] = a
        output_buffer[j+1] = b
        output_buffer[j+2] = c
        output_buffer[j+3] = d

        j += 4

    return output_buffer
