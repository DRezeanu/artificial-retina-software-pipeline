//
// Created by Eric Wu on 4/8/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include "visionload_extensions.h"

namespace py = pybind11;

PYBIND11_MODULE(visionload_cpp_extensions, m) {

    m.doc() = "C++ extensions for visionloader"; // optional module docstring


    m.def("unpack_rgb_sta",
            &unpack_rgb_sta,
            pybind11::return_value_policy::take_ownership,
            "Unpack char buffer for color STA for reading");
}
