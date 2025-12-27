//
// Created by Eric Wu on 4/8/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>

#include "RandomFrame.h"

namespace py = pybind11;

PYBIND11_MODULE(noise_frame_generator, m) {

    m.doc() = "Generate frames for white noise"; // optional module docstring

    py::class_<JavaRandSequence>(m, "JavaRandSequence")
    .def(py::init<int32_t>())
    .def("advance", &JavaRandSequence::advance)
    .def("randJavaNbit", &JavaRandSequence::randJavaNbit)
    .def("randJavaLong", &JavaRandSequence::randJavaLong)
    .def("randJavaFloat", &JavaRandSequence::randJavaFloat)
    .def(py::pickle(
        [](const JavaRandSequence &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.state, p._seed);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 2)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            JavaRandSequence p(t[0].cast<int64_t>(), t[1].cast<int64_t>());

            return p;
        }
    ));

    m.def("draw_random_single_frame",
            &draw_random_single_frame,
            pybind11::return_value_policy::take_ownership,
            "Draw single white noise frame");

    m.def("draw_consecutive_frames",
            &draw_consecutive_frames,
            pybind11::return_value_policy::take_ownership,
            "Generate a series of frames");

	m.def("draw_upsampled_jittered_consecutive_frames",
			&draw_upsampled_jittered_consecutive_frames,
			pybind11::return_value_policy::take_ownership,
			"Generate a series of randomly jittered frames");

	m.def("draw_upsampled_jittered_frame",
			&draw_upsampled_jittered_frame,
			pybind11::return_value_policy::take_ownership,
			"Generate a single jittered frame");

    m.def("advance_seed_n_frames",
          &advance_seed_n_frames);
}

