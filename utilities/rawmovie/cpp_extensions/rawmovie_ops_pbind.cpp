#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>

#include "rawmovie_ops.h"

namespace py = pybind11;

PYBIND11_MODULE(rawmovie_ops, m) {

    m.doc() = "Optimized routines for handling raw movies";

    m.def("convert_color_to_bw_float32",
            &convert_color_to_bw_float32,
            pybind11::return_value_policy::take_ownership,
            "Convert batch of color uint8_t frames to BW float32");

    m.def("convert_color_8bit_to_bw_float32_noncontig",
            &convert_color_8bit_to_bw_float32_noncontig,
            pybind11::return_value_policy::take_ownership,
            "Convert batch of color uint8_t frames to BW float32, with reduced copying at the input");
}

