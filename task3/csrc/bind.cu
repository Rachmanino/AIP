#include<ndarray.h>
#include<fn.h>

PYBIND11_MODULE(backend, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<NDArray>(m, "NDArray")
        .def(py::init<shape_t, Device>())
        .def(py::init<shape_t, scalar_t, Device>())
        .def(py::init<const NDArray&>())
        .def_readonly("shape", &NDArray::shape)
        .def_readonly("size", &NDArray::size)
        .def_readonly("device", &NDArray::device)
        .def("from_array", &NDArray::from_array)
        .def("tolist", &NDArray::tolist)
        .def("rand", &NDArray::rand)
        .def("randn", &NDArray::randn)
        .def("__add__", &NDArray::operator +)
        .def("__add__", &add_scalar)
        .def("__sub__", &NDArray::operator -)
        .def("__sub__", &sub_scalar)
        .def("__mul__", &mul)
        .def("__mul__", &mul_scalar)
        .def("__matmul__", &NDArray::operator *)
        .def("__truediv__", &NDArray::operator /)
        .def("__truediv__", &div_scalar)
        .def("__pow__", &pow_scalar)
        .def("cpu", &NDArray::cpu)
        .def("gpu", &NDArray::gpu)
        .def("to", &NDArray::to)
        .def("reshape", &NDArray::reshape)
        .def("T", &NDArray::T2d)
        .def("swap", &NDArray::swap);
    
    m.def_submodule("fn")
        .def("fc_fwd", &fc_fwd)
        .def("fc_bwd", &fc_bwd)
        .def("conv2d_k33p1s1_fwd", &conv2d_k33p1s1_fwd)
        .def("conv2d_k33p1s1_bwd", &conv2d_k33p1s1_bwd)
        .def("maxpooling2d_k22s2_fwd", &maxpooling2d_k22s2_fwd)
        .def("maxpooling2d_k22s2_bwd", &maxpooling2d_k22s2_bwd)
        .def("softmax_fwd", &softmax_fwd)
        .def("celoss_fwd", &celoss_fwd)
        .def("softmax_ce_bwd", &softmax_ce_bwd)
        .def("relu_fwd", &relu_fwd)
        .def("relu_bwd", &relu_bwd)
        .def("sigmoid_fwd", &sigmoid_fwd)
        .def("sigmoid_bwd", &sigmoid_bwd);
}
