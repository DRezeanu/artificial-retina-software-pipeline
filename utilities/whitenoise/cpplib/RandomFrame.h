#include <string>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define RANDJAVA_SCALE (4294967296.0)

namespace py=pybind11;

class JavaRandSequence {

public:
    int64_t state;
    int64_t _seed;

    JavaRandSequence(int64_t old_state, int64_t old_seed) {
        /*
        Don't use this constructor in normal use; this is only to
            restore the state of a pickled JavaRandSequence

        If you try to manually set the seed and state together
            you'll probably get total noncsense
        */

        state = old_state;
        _seed = old_seed;
    }

    JavaRandSequence(int64_t seed) {
	    _seed = seed;
        state = (seed ^ 0x5DEECE66DLL) & 0xFFFFFFFFFFFFLL;
    }

    void reset_to_beginning() {
        state = (_seed ^ 0x5DEECE66DLL) & 0xFFFFFFFFFFFFLL;
    }

    void advance(int64_t num_to_advance) {
        for (int64_t i = 0; i < num_to_advance; ++i) {
            state = (state * 0x5DEECE66DLL + 0xBLL) & 0xFFFFFFFFFFFFLL;
        }
    }

    int16_t randJavaNbit(int64_t n_bit) {

        state = (state * 0x5DEECE66DLL + 0xBLL) & 0xFFFFFFFFFFFFLL;

        if (n_bit == 1) {
            return (int16_t)(state >> 47LL);
        } else if (n_bit == 3) {
            return (int16_t)(state >> 45LL);
        } else if (n_bit == 8) {
            return (int16_t)(state >> 40LL);
        } else {
            return (int16_t)(state >> 32LL);
        }

    }

    int16_t randJavaShort() {
        state = (state * 0x5DEECE66DLL + 0xBLL) & 0xFFFFFFFFFFFFLL;
        return (int16_t)(state >> 32LL);
    }

    uint16_t randJavaUShort() {
        state = (state * 0x5DEECE66DLL + 0xBLL) & 0xFFFFFFFFFFFFLL;
        return (uint16_t)(state >> 32LL);
    }

    int32_t randJavaLong() {
        state = (state * 0x5DEECE66DLL + 0xBLL) & 0xFFFFFFFFFFFFLL;
        return (int32_t)(state >> 16LL);
    }

    float randJavaFloat() {
        return (((float) (uint16_t)(randJavaLong() >> 16LL)) / 65535.0f);
    }
};


py::array_t<uint8_t, py::array::c_style | py::array::forcecast> draw_consecutive_frames(
        JavaRandSequence &rng,
        int32_t width, // width of the output array
        int32_t height, // height of the output array
        int32_t n_frames,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &lut_np,
        py::array_t<uint16_t, py::array::c_style | py::array::forcecast> &map_np,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &back_rgb_np,
        int32_t m_width, // width in stixels
        int32_t m_height, // height in stixels
        int32_t noise_type,
        int32_t n_bits,
        float probability) {

    /*
     * The main change here compared to Photons is that we have to change the array order
     * because MATLAB defaults to Fortran order while default numpy/C++
     * prefers C order
     *
     * In this case, the output shape goes from (4, width, height)
     * to (n_frames, height, width, 3)
     * We also ditch the luminance channel because we don't care
     */

    py::buffer_info lut_info = lut_np.request();
    uint8_t *lut = static_cast<uint8_t *> (lut_info.ptr);

    // unpack map_np, grab pointer
    // note that the array might be (0, 0) in shape
    // in which case we don't want to use the map at all
    py::buffer_info map_info = map_np.request();
    uint16_t *map = nullptr;
    if (map_info.shape[0] != 0) {
        map = static_cast<uint16_t *> (map_info.ptr);
    }

    // unpack back_rgb_np, grab pointer
    py::buffer_info back_rgb_info = back_rgb_np.request();
    uint8_t *backrgb = static_cast<uint8_t *> (back_rgb_info.ptr);

    // create numpy arrays for the output
    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(uint8_t),     /* Size of one item */
            py::format_descriptor<uint8_t>::value, /* Buffer format */
            4,          /* How many dimensions? */
            {n_frames, height, width, 3},  /* Number of elements for each dimension */
            {sizeof(uint8_t) * 3 * width * height, sizeof(uint8_t) * 3 * width, sizeof(uint8_t) * 3, sizeof(uint8_t)}
            /* Strides for each dimension */
    );

    py::array_t <uint8_t> output_np_array = py::array_t<uint8_t>(output_buffer_info);
    py::buffer_info output_info = output_np_array.request();
    uint8_t *output_buffer = static_cast<uint8_t *> (output_info.ptr);

    int32_t h, w, cnt;
    int32_t if_fill = 1;
    int32_t image_index, lut_index;
    int32_t map_index, map_value;

    uint8_t *prefilled_seq = new uint8_t[m_width * m_height * 3]; // note the type change from the original
    uint8_t *image_pattern = new uint8_t[width * height * 3]; // extra buffer to make refactoring easy

    uint64_t memcopy_offset;

    for (int32_t frame_idx = 0; frame_idx < n_frames; ++frame_idx) {

        memcopy_offset = frame_idx * height * width * 3;

        for (h = 0; h < m_height; ++h) {

            image_index = 3 * h * m_width;
            for (w = m_width; w != 0; --w) {

                if (probability != 1.0) {
                    if_fill = rng.randJavaFloat() < probability;
                }

                if (if_fill) { // fill color values
                    for (cnt = 0; cnt < 3; cnt++) {
                        if (noise_type == 3 || cnt == 0) { // Gaussian RGB - 2 additional draws
                            lut_index = (int32_t)(rng.randJavaNbit(n_bits) * 3);
                            // stateVal is the copy of the seed (pointer) taking in the parameter rng_state,
                            // which is the value of the seed
                        }
                        prefilled_seq[image_index++] = lut[lut_index + cnt];
                    }
                } else { // fill background values
                    for (cnt = 0; cnt < 3; cnt++) {
                        prefilled_seq[image_index++] = backrgb[cnt];
                    }
                }
            }
        }

        if (map != nullptr) { // condition evaluates to False if None

            // unpack mat_np, grab pointer, since we now know it exists
            py::buffer_info map_info = map_np.request();
            uint16_t *map = static_cast<uint16_t *> (map_info.ptr);

            for (h = 0; h < height; ++h) {
                image_index = 3 * h * width;
                map_index = h * width;
                for (w = 0; w < width; ++w) {
                    map_value = (int32_t) map[map_index++];

                    if (map_value > 0) { //'cone'
                        cnt = (map_value - 1) * 3;
                        image_pattern[image_index++] = prefilled_seq[cnt++];
                        image_pattern[image_index++] = prefilled_seq[cnt++];
                        image_pattern[image_index++] = prefilled_seq[cnt];

                    } else { // intercone space    }
                        image_pattern[image_index++] = backrgb[0];  //  R
                        image_pattern[image_index++] = backrgb[1];  // G
                        image_pattern[image_index++] = backrgb[2];  // B
                    }
                }
            }
        } else {
            // copy stuff over to array
            memcpy(image_pattern, prefilled_seq, sizeof(uint8_t) * 3 * width * height);
        }

        memcpy(output_buffer + memcopy_offset, image_pattern, sizeof(uint8_t) * 3 * width * height);
    }

    delete[] image_pattern;
    delete[] prefilled_seq;

    return output_np_array;

}

py::array_t<uint8_t, py::array::c_style | py::array::forcecast> draw_upsampled_jittered_consecutive_frames(
        JavaRandSequence &stixel_val_rng,
        JavaRandSequence &jitter_rng,
        int32_t width,
        int32_t height,
        int32_t n_frames,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &lut_np,
        py::array_t<uint16_t, py::array::c_style | py::array::forcecast> &map_np,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &back_rgb_np,
        int32_t m_width, // width in stixels
        int32_t m_height, // height in stixels
        int32_t noise_type,
        int32_t n_bits,
        float probability,
        int32_t stixel_width,
        int32_t stixel_height) {

    // unjittered_buffer has shape (n_frames, height, width, 3)
    auto unjittered_buffer = draw_consecutive_frames(
            stixel_val_rng,
            width,
            height,
            n_frames,
            lut_np,
            map_np,
            back_rgb_np,
            m_width,
            m_height,
            noise_type,
            n_bits,
            probability
    );

	py::buffer_info unjittered_info = unjittered_buffer.request();
	uint8_t *read_buffer = static_cast<uint8_t *>(unjittered_info.ptr);

    int32_t pixel_height = height * stixel_height;
    int32_t pixel_width = width * stixel_width;
    int16_t half_stixel_x = (stixel_width >> 1);
    int16_t half_stixel_y = (stixel_height >> 1);

    int32_t stixel_width_log2 = 0;
    int32_t stixel_height_log2 = 0;
    int32_t temp_stixel_width = stixel_width;
    int32_t temp_stixel_height = stixel_height;

    while (temp_stixel_height >>= 1) ++stixel_height_log2;
    while (temp_stixel_width >>= 1) ++stixel_width_log2;

    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(uint8_t),     /* Size of one item */
            py::format_descriptor<uint8_t>::value, /* Buffer format */
            4,          /* How many dimensions? */
            {n_frames, pixel_height, pixel_width, 3},  /* Number of elements for each dimension */
            {sizeof(uint8_t) * 3 * pixel_height * pixel_width, sizeof(uint8_t) * 3 * pixel_width, sizeof(uint8_t) * 3,
             sizeof(uint8_t)}
            /* Strides for each dimension */
    );

    py::array_t <uint8_t> output_np_array = py::array_t<uint8_t>(output_buffer_info);
	py::buffer_info output_info = output_np_array.request();
	uint8_t *output_buffer = static_cast<uint8_t *>(output_info.ptr);

    int16_t jitter_x, jitter_y;
    int64_t frame_write_offset, frame_read_offset;

    bool in_bounds_x, in_bounds_y;
    int32_t shifted_pixel_x, shifted_pixel_y;
    int32_t orig_stixel_x, orig_stixel_y;

    for (int32_t frame_counter = 0; frame_counter < n_frames; ++frame_counter) {

        jitter_x = static_cast<int16_t>(jitter_rng.randJavaUShort() % stixel_width) - half_stixel_x;
        jitter_y = static_cast<int16_t > (jitter_rng.randJavaUShort() % stixel_height) - half_stixel_y;

		frame_write_offset = frame_counter * pixel_height * pixel_width * 3;
		frame_read_offset = frame_counter * height * width * 3;

        // now copy shifted/upsampled data to the proper output buffer
        for (int32_t pixel_y = 0; pixel_y < pixel_height; ++pixel_y) {
            for (int32_t pixel_x = 0; pixel_x < pixel_width; ++pixel_x) {
                // figure which original stixel goes here
                shifted_pixel_x = pixel_x - jitter_x;
                shifted_pixel_y = pixel_y - jitter_y;

                orig_stixel_x = (shifted_pixel_x) >> stixel_width_log2;
                orig_stixel_y = (shifted_pixel_y) >> stixel_height_log2;

                in_bounds_x = (shifted_pixel_x >= 0) && (shifted_pixel_x < pixel_width);
                in_bounds_y = (shifted_pixel_y >= 0) && (shifted_pixel_y < pixel_height);

                if (in_bounds_x && in_bounds_y) {
                    for (int32_t channel_ix = 0; channel_ix < 3; ++channel_ix) {
						int64_t write_idx = frame_write_offset + (pixel_y * pixel_width * 3) + pixel_x * 3 + channel_ix;
						int64_t read_idx = frame_read_offset + (orig_stixel_y * width * 3) + orig_stixel_x * 3 + channel_ix;
						*(output_buffer + write_idx) = *(read_buffer + read_idx);
                    }
                } else {
                    for (int32_t channel_ix = 0; channel_ix < 3; ++channel_ix) {
						int64_t write_idx = frame_write_offset + (pixel_y * pixel_width * 3) + pixel_x * 3 + channel_ix;
						*(output_buffer + write_idx) = 0;
                    }
                }
            }
        }

    }

    return output_np_array;
}


void advance_seed_n_frames(
        JavaRandSequence &rng,
        int32_t m_width, // width in stixels
        int32_t m_height, // height in stixels
        int32_t noise_type,
        int32_t n_bits,
        float probability,
        int32_t n_frames) {

    /*
    Advance the number of frames without touching memory, faster than generating all of the frames
        and committing to main memory, and then ignoring the result
    */

    int32_t cnt;
    int32_t if_fill = 1;

    int64_t n_stixels_total = m_width * m_height * n_frames;

    for (int64_t frame_idx = 0; frame_idx < n_stixels_total; ++frame_idx) {
        if (probability != 1.0) {
            if_fill = rng.randJavaFloat() < probability;
        }

        if (if_fill) { // fill color values
            for (cnt = 0; cnt < 3; cnt++) {
                if (noise_type == 3 || cnt == 0) { // Gaussian RGB - 2 additional draws
                    rng.randJavaNbit(n_bits);
                }
            }
        }
    }
}


py::array_t<uint8_t, py::array::c_style | py::array::forcecast> draw_random_single_frame(
        JavaRandSequence &rng,
        int32_t width, // width in terms of stixels
        int32_t height, // height in terms of stixels
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &lut_np,
        py::array_t<uint16_t, py::array::c_style | py::array::forcecast> &map_np,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &back_rgb_np,
        int32_t m_width, // width of map if it exists, otherwise width in stixels
        int32_t m_height, // height of map if it exists, otherwise height in stixels
        int32_t noise_type,
        int32_t n_bits,
        float probability) {

    /*
     * The main change here compared to Photons is that we have to change the array order
     * because MATLAB defaults to Fortran order while default numpy/C++
     * prefers C order
     *
     * In this case, the output shape goes from (4, width, height)
     * to (height, width, 3)
     */

    // unpack the inputs

    // unpack lut_np, grab pointer
    py::buffer_info lut_info = lut_np.request();
    uint8_t *lut = static_cast<uint8_t *> (lut_info.ptr);

    // unpack map_np, grab pointer
    // note that the array might be (0, 0) in shape
    // in which case we don't want to use the map at all
    py::buffer_info map_info = map_np.request();
    uint16_t *map = nullptr;
    if (map_info.shape[0] != 0) {
        map = static_cast<uint16_t *> (map_info.ptr);
    }

    // unpack back_rgb_np, grab pointer
    py::buffer_info back_rgb_info = back_rgb_np.request();
    uint8_t *backrgb = static_cast<uint8_t *> (back_rgb_info.ptr);

    // create numpy arrays for the output
    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(uint8_t),     /* Size of one item */
            py::format_descriptor<uint8_t>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {height, width, 3},  /* Number of elements for each dimension */
            {sizeof(uint8_t) * 3 * width, sizeof(uint8_t) * 3, sizeof(uint8_t)}  /* Strides for each dimension */
    );

    py::array_t <uint8_t> output_np_array = py::array_t<uint8_t>(output_buffer_info);
    py::buffer_info output_info = output_np_array.request();
    uint8_t *image_pattern = static_cast<uint8_t *> (output_info.ptr);

    int32_t h, w, cnt;
    int32_t if_fill = 1;
    int32_t image_index, lut_index;
    int32_t map_index, map_value;

    uint8_t *prefilled_seq = new uint8_t[m_width * m_height * 3]; // note the type change from the original

    // This loop packs prefilled_seq
    for (h = 0; h < m_height; ++h) { // loop over either height of the map if it exists or the stixel height

        image_index = 3 * h * m_width;
        for (w = m_width; w != 0; --w) { // loop over either the width of the map if it exists or the stixel width

            if (probability != 1.0)
                if_fill = rng.randJavaFloat() < probability;

            if (if_fill) { // fill color values
                for (cnt = 0; cnt < 3; cnt++) {
                    if (noise_type == 3 || cnt == 0) { // Gaussian RGB - 2 additional draws
                        lut_index = (int32_t)(rng.randJavaNbit(n_bits) * 3);
                        // stateVal is the copy of the seed (pointer) taking in the parameter rng_state,
                        // which is the value of the seed
                    }
                    prefilled_seq[image_index++] = lut[lut_index + cnt];
                }
            } else { // fill background values
                for (cnt = 0; cnt < 3; cnt++) {
                    prefilled_seq[image_index++] = backrgb[cnt];
                }
            }
        }
    }

    // in the simple white noise case we do not hit this condition
    if (map != nullptr) { // condition evaluates to False if None

        // unpack mat_np, grab pointer, since we now know it exists
        py::buffer_info map_info = map_np.request();
        uint16_t *map = static_cast<uint16_t *> (map_info.ptr);

        // loop over the height of the output image, in pixels
        for (h = 0; h < height; ++h) {
            image_index = 3 * h * width;
            map_index = h * width;

            // loop over the width of the output image, in pixels
            for (w = 0; w < width; ++w) {
                map_value = (int32_t) map[map_index++];

                if (map_value > 0) { //'cone'
                    cnt = (map_value - 1) * 3;
                    image_pattern[image_index++] = prefilled_seq[cnt++];
                    image_pattern[image_index++] = prefilled_seq[cnt++];
                    image_pattern[image_index++] = prefilled_seq[cnt];

                } else { // intercone space
                    image_pattern[image_index++] = backrgb[0];  //  R
                    image_pattern[image_index++] = backrgb[1];  // G
                    image_pattern[image_index++] = backrgb[2];  // B
                }
            }
        }
    } else {
        // copy stuff over to array
        memcpy(image_pattern, prefilled_seq, sizeof(uint8_t) * 3 * width * height);
    }

    delete[] prefilled_seq;

    return output_np_array;
}


py::array_t<uint8_t, py::array::c_style | py::array::forcecast> draw_upsampled_jittered_frame(
        JavaRandSequence &stixel_val_rng,
        JavaRandSequence &jitter_rng,
        int32_t width,
        int32_t height,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &lut_np,
        py::array_t<uint16_t, py::array::c_style | py::array::forcecast> &map_np,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &back_rgb_np,
        int32_t m_width, // width in stixels
        int32_t m_height, // height in stixels
        int32_t noise_type,
        int32_t n_bits,
        float probability,
        int32_t stixel_width,
        int32_t stixel_height) {

    // has shape (height, width, 3)
    auto unjittered_single_frame = draw_random_single_frame(
            stixel_val_rng,
            width,
            height,
            lut_np,
            map_np,
            back_rgb_np,
            m_width,
            m_height,
            noise_type,
            n_bits,
            probability
    );

	py::buffer_info unjittered_info = unjittered_single_frame.request();
	uint8_t *read_buffer = static_cast<uint8_t *>(unjittered_info.ptr);

    int32_t pixel_height = height * stixel_height;
    int32_t pixel_width = width * stixel_width;

    int16_t half_stixel_x = (stixel_width >> 1);
    int16_t half_stixel_y = (stixel_height >> 1);

    int32_t stixel_width_log2 = 0;
    int32_t stixel_height_log2 = 0;
    int32_t temp_stixel_width = stixel_width;
    int32_t temp_stixel_height = stixel_height;

    while (temp_stixel_height >>= 1) ++stixel_height_log2;
    while (temp_stixel_width >>= 1) ++stixel_width_log2;

    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(uint8_t),     /* Size of one item */
            py::format_descriptor<uint8_t>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {pixel_height, pixel_width, 3},  /* Number of elements for each dimension */
            {sizeof(uint8_t) * 3 * pixel_width, sizeof(uint8_t) * 3, sizeof(uint8_t)}
            /* Strides for each dimension */
    );

    py::array_t <uint8_t> output_np_array = py::array_t<uint8_t>(output_buffer_info);
	py::buffer_info output_info = output_np_array.request();
	uint8_t *output_buffer = static_cast<uint8_t *>(output_info.ptr);

    int16_t jitter_x = static_cast<int16_t>(jitter_rng.randJavaUShort() % stixel_width) - half_stixel_x;
    int16_t jitter_y = static_cast<int16_t >(jitter_rng.randJavaUShort() % stixel_height) - half_stixel_y;

    int32_t shifted_pixel_x, shifted_pixel_y;
    int32_t orig_stixel_x, orig_stixel_y;
    bool in_bounds_x, in_bounds_y;


    // now copy shifted/upsampled data to the proper output buffer
    for (int32_t pixel_y = 0; pixel_y < pixel_height; ++pixel_y) {
        for (int32_t pixel_x = 0; pixel_x < pixel_width; ++pixel_x) {
            // figure which original stixel goes here
            shifted_pixel_x = pixel_x - jitter_x;
            shifted_pixel_y = pixel_y - jitter_y;

            orig_stixel_x = (shifted_pixel_x) >> stixel_width_log2;
            orig_stixel_y = (shifted_pixel_y) >> stixel_height_log2;

            in_bounds_x = (shifted_pixel_x >= 0) && (shifted_pixel_x < pixel_width);
            in_bounds_y = (shifted_pixel_y >= 0) && (shifted_pixel_y < pixel_height);

            if (in_bounds_x && in_bounds_y) {
                for (int32_t channel_ix = 0; channel_ix < 3; ++channel_ix) {
					int32_t write_idx = (pixel_y * pixel_width * 3) + (pixel_x * 3) + channel_ix;
					int32_t read_idx = (orig_stixel_y * width * 3) + (orig_stixel_x * 3) + channel_ix;
					*(output_buffer + write_idx) = *(read_buffer + read_idx);
                }
            } else {
                for (int32_t channel_ix = 0; channel_ix < 3; ++channel_ix) {
					int32_t write_idx = (pixel_y * pixel_width * 3) + (pixel_x * 3) + channel_ix;
					*(output_buffer + write_idx) = 0;
                }
            }
        }
    }

    return output_np_array;
}

