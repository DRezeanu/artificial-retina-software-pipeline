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


    // No return_value_policy: both functions return py::bytes by value, which is already
    // a Python object. Ownership policies govern returned pointers and references to C++
    // objects; for py::bytes there is nothing for pybind11 to take ownership of, and the
    // policy was ignored. The buffers are freed explicitly with delete[] before returning.
    m.def("pack_sta_buffer_color",
            &pack_sta_buffer_color,
            "Pack char buffer with color STA for writing");

    m.def("pack_ei_matrices",
            &pack_ei_matrices,
            "Pack char buffer with EI and EI error for writing");


}