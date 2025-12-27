#include <string>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>

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

std::tuple<py::array_t<float, py::array::c_style | py::array::forcecast>,
           py::array_t<float, py::array::c_style | py::array::forcecast>,
           py::array_t<float, py::array::c_style | py::array::forcecast>,
           py::array_t<float, py::array::c_style | py::array::forcecast>,
           py::array_t<float, py::array::c_style | py::array::forcecast>,
           py::array_t<float, py::array::c_style | py::array::forcecast>> unpack_rgb_sta (
    const char *raw_buffer,
    size_t sta_width,
    size_t sta_height,
    size_t sta_depth
) {

    // create all of the empty numpy arrays
    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(float),     /* Size of one item */
            py::format_descriptor<float>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {sta_depth, sta_width, sta_height},  /* Number of elements for each dimension */
            {sizeof(float) * sta_width * sta_height, sizeof(float) * sta_height, sizeof(float)}
            /* Strides for each dimension */
    );

    py::array_t <float> red_data = py::array_t<float>(output_buffer_info);
    py::buffer_info output_info = red_data.request();
    float *red_data_buffer = static_cast<float *> (output_info.ptr);

    py::array_t <float> red_error = py::array_t<float>(output_buffer_info);
    output_info = red_error.request();
    float *red_err_buffer = static_cast<float *> (output_info.ptr);

    py::array_t <float> green_data = py::array_t<float>(output_buffer_info);
    output_info = green_data.request();
    float *green_data_buffer = static_cast<float *> (output_info.ptr);

    py::array_t <float> green_error = py::array_t<float>(output_buffer_info);
    output_info = green_error.request();
    float *green_err_buffer = static_cast<float *> (output_info.ptr);

    py::array_t <float> blue_data = py::array_t<float>(output_buffer_info);
    output_info = blue_data.request();
    float *blue_data_buffer = static_cast<float *> (output_info.ptr);

    py::array_t <float> blue_error = py::array_t<float>(output_buffer_info);
    output_info = blue_error.request();
    float *blue_err_buffer = static_cast<float *> (output_info.ptr);

    size_t dwh_offset, wh_offset, write_offset;
    size_t read_offset = 3; // need to offset one 32 bit integer don't cares, and one 64 bit double don't care

    const uint32_t *raw_buffer_as_int = reinterpret_cast<const uint32_t *>(raw_buffer);
    uint32_t temp;
    for (size_t i = 0; i < sta_depth; ++i) {

	read_offset += 4; // need to offset two 32 bit integers and one 64 bit double that we don't care about

        dwh_offset = i * sta_width * sta_height;

        for (size_t j = 0; j < sta_width; ++j) {

            wh_offset = dwh_offset + j * sta_height;

            for (size_t k = 0; k < sta_height; ++k) {

                write_offset = wh_offset + k;

                temp = bswap32(*(raw_buffer_as_int+read_offset));
                *(red_data_buffer+write_offset) = *(reinterpret_cast<float *>(&temp));
                ++read_offset;

                temp = bswap32(*(raw_buffer_as_int+read_offset));
                *(red_err_buffer+write_offset) = *(reinterpret_cast<float *>(&temp));
                ++read_offset;

                temp = bswap32(*(raw_buffer_as_int+read_offset));
                *(green_data_buffer+write_offset) = *(reinterpret_cast<float *>(&temp));
                ++read_offset;

                temp = bswap32(*(raw_buffer_as_int+read_offset));
                *(green_err_buffer+write_offset) = *(reinterpret_cast<float *>(&temp));
                ++read_offset;

                temp = bswap32(*(raw_buffer_as_int+read_offset));
                *(blue_data_buffer+write_offset) = *(reinterpret_cast<float *>(&temp));
                ++read_offset;

                temp = bswap32(*(raw_buffer_as_int+read_offset));
                *(blue_err_buffer+write_offset) = *(reinterpret_cast<float *>(&temp));
                ++read_offset;
            }
        }
    }

    return std::make_tuple(red_data, red_error, green_data, green_error, blue_data, blue_error);
}
