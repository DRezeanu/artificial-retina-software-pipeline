//
// Created by Eric Wu on 4/8/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include "visionwrite_extensions.h"

namespace py = pybind11;

PYBIND11_MODULE(visionwrite_cpp_extensions, m) {

    m.doc() = "C++ extensions for visionwriter"; // optional module docstring


    m.def("pack_sta_buffer_color",
            &pack_sta_buffer_color,
            pybind11::return_value_policy::take_ownership,
            "Pack char buffer with color STA for writing");

    m.def("pack_ei_matrices",
            &pack_ei_matrices,
            pybind11::return_value_policy::take_ownership,
            "Pack char buffer with EI and EI error for writing");


}