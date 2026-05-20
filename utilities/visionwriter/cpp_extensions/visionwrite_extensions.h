#include <string>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


#if defined(__has_builtin)
  #if __has_builtin(__builtin_bswap32)
    #define HAS_BUILTIN_BSWAP32 1
  #endif
  #if __has_builtin(__builtin_bswap64)
    #define HAS_BUILTIN_BSWAP64 1
  #endif
#endif

static inline uint32_t bswap32(uint32_t x) {
#if defined(HAS_BUILTIN_BSWAP32)
    return __builtin_bswap32(x);
#elif defined(_MSC_VER)
    return _byteswap_ulong(x);
#else
    return (x >> 24) |
           ((x >> 8) & 0x0000FF00u) |
           ((x << 8) & 0x00FF0000u) |
           (x << 24);
#endif
}

static inline uint64_t bswap64(uint64_t x) {
#if defined(HAS_BUILTIN_BSWAP64)
    return __builtin_bswap64(x);
#elif defined(_MSC_VER)
    return _byteswap_uint64(x);
#else
    return (x >> 56) |
           ((x >> 40) & 0x000000000000FF00ull) |
           ((x >> 24) & 0x0000000000FF0000ull) |
           ((x >> 8)  & 0x00000000FF000000ull) |
           ((x << 8)  & 0x000000FF00000000ull) |
           ((x << 24) & 0x0000FF0000000000ull) |
           ((x << 40) & 0x00FF000000000000ull) |
           (x << 56);
#endif
}


namespace py=pybind11;

py::bytes pack_ei_matrices(
    py::array_t<float, py::array::c_style | py::array::forcecast>& ei_matrix,
    py::array_t<float, py::array::c_style | py::array::forcecast>& ei_error) {

    py::buffer_info ei_info = ei_matrix.request();
    float *ei_data_ptr = static_cast<float *> (ei_info.ptr);

    size_t n_els = ei_info.shape[0];
    size_t n_samples = ei_info.shape[1];

    py::buffer_info error_info = ei_error.request();
    float *error_ptr = static_cast<float *>(error_info.ptr);

    size_t n_output_entries = (1 + n_els) * n_samples * 2;
    uint32_t *output_buffer= new uint32_t[n_output_entries];

    size_t write_offset = 0;

    // first have to write the TTL nonsense channels
    // since we assume that ei_matrix and ei_error correspond only to real electrodes
    for (size_t j = 0; j < n_samples; ++j) {
        output_buffer[write_offset++] = 0;
        output_buffer[write_offset++] = 0;
    }

    for (size_t i = 0; i < n_els; ++i) {
        for (size_t j = 0; j < n_samples; ++j) {
            size_t read_ix = i * n_samples + j;
            output_buffer[write_offset++] = bswap32(*(reinterpret_cast<uint32_t *>(ei_data_ptr + read_ix)));
            output_buffer[write_offset++] = bswap32(*(reinterpret_cast<uint32_t *>(error_ptr + read_ix)));
        }
    }

    return py::bytes(reinterpret_cast<char *> (output_buffer), n_output_entries * 4);
}

py::bytes pack_sta_buffer_color (
    py::array_t<float, py::array::c_style | py::array::forcecast>& red_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& red_err,
    py::array_t<float, py::array::c_style | py::array::forcecast>& green_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& green_err,
    py::array_t<float, py::array::c_style | py::array::forcecast>& blue_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& blue_err,
    double stixel_size) {

    py::buffer_info red_sta_info = red_sta.request();
    float *red_data_ptr = static_cast<float *> (red_sta_info.ptr);

    size_t sta_depth = red_sta_info.shape[0];
    size_t sta_width = red_sta_info.shape[1];
    size_t sta_height = red_sta_info.shape[2];

    size_t n_output_entries = 6 * sta_width * sta_height * sta_depth + sta_depth * 4;

    py::buffer_info red_err_info = red_err.request();
    float *red_err_ptr = static_cast<float *> (red_err_info.ptr);

    py::buffer_info green_sta_info = green_sta.request();
    float *green_data_ptr = static_cast<float *> (green_sta_info.ptr);

    py::buffer_info green_err_info = green_err.request();
    float *green_err_ptr = static_cast<float *> (green_err_info.ptr);

    py::buffer_info blue_sta_info = blue_sta.request();
    float *blue_data_ptr = static_cast<float *> (blue_sta_info.ptr);

    py::buffer_info blue_err_info = blue_err.request();
    float *blue_err_ptr = static_cast<float *> (blue_err_info.ptr);

    uint32_t *output_buffer = new uint32_t[n_output_entries];

    size_t depth_offset, width_depth_offset, read_offset;
    uint64_t stixel_temp;
    size_t write_idx = 0;
    for (size_t i = 0; i < sta_depth; ++i) {

        depth_offset = i * (sta_width * sta_height);

        output_buffer[write_idx++] = bswap32(static_cast<uint32_t>(sta_height));
        output_buffer[write_idx++] = bswap32(static_cast<uint32_t>(sta_width));

        stixel_temp = bswap64(*(reinterpret_cast<uint64_t *>(&stixel_size)));
        output_buffer[write_idx++] = static_cast<uint32_t> (stixel_temp >> 32);
        output_buffer[write_idx++] = static_cast<uint32_t> (stixel_temp & 0xFFFF);
        
        // TODO: Eventually replace above with code below to aviod undefined behavior (strict aliasing)
        // uint64_t refresh_bits;
        //std::memcpy(&refresh_bits, &refresh_time, sizeof(refresh_bits));
        //refresh_bits = bswap64(refresh_bits);
        //output_buffer[write_idx++] = static_cast<uint32_t>(refresh_bits >> 32);
        //output_buffer[write_idx++] = static_cast<uint32_t>(refresh_bits & 0xFFFFFFFFu);

        for (size_t j = 0; j < sta_width; ++j)  {

            width_depth_offset = j * sta_height + depth_offset;

            for (size_t k = 0; k < sta_height; ++k) {

                read_offset = width_depth_offset + k;

                output_buffer[write_idx++] = bswap32(*(reinterpret_cast<uint32_t *> (red_data_ptr + read_offset)));
                output_buffer[write_idx++] = bswap32(*(reinterpret_cast<uint32_t *> (red_err_ptr + read_offset)));

                output_buffer[write_idx++] = bswap32(*(reinterpret_cast<uint32_t *> (green_data_ptr + read_offset)));
                output_buffer[write_idx++] = bswap32(*(reinterpret_cast<uint32_t *> (green_err_ptr + read_offset)));

                output_buffer[write_idx++] = bswap32(*(reinterpret_cast<uint32_t *> (blue_data_ptr + read_offset)));
                output_buffer[write_idx++] = bswap32(*(reinterpret_cast<uint32_t *> (blue_err_ptr + read_offset)));

            }
        }
    }


    // TODO: copy output buffer to pybytes and then delete it before returning (more secure, free up memory once used)
    // py::bytes result(reinterpret_cast<char *> (output_buffer), n_output_entries * 4);
    // delete[] output_buffer;
    // return result
    return py::bytes(reinterpret_cast<char *> (output_buffer), n_output_entries * 4);
}
