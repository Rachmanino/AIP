#ifndef NDARRAY_H
#define NDARRAY_H

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>

#include<cuda.h>
#include<thrust/for_each.h>
#include<thrust/reduce.h>
#include<thrust/device_vector.h>
#include<cublas_v2.h>
#include<curand.h>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

#define MAX_DIMS 100 // Max # dimensions

namespace py = pybind11;
typedef float scalar_t; // Currently only support float type
typedef std::vector<float> list_t; // List[float] in Python
typedef std::vector<int> shape_t;   // List[int] in Python
typedef py::array_t<scalar_t> array_t;  // numpy array

enum Device {
    CPU,
    GPU
};

class NDArray {
    public:
        Device device;
        scalar_t* p; // pointer to real data
        shape_t shape;
        int size;

        NDArray(const shape_t shape, Device device);
        NDArray(const shape_t shape, scalar_t val, Device device);
        static NDArray& from_array(const array_t& src, Device device); // from np.ndarray
        list_t& tolist(); // to Python list

        static NDArray& rand(const shape_t shape, scalar_t l, scalar_t h, Device device);
        static NDArray& randn(const shape_t shape, scalar_t mean, scalar_t std, Device device);

        NDArray(const NDArray& src); // deep copy
        NDArray& operator = (const NDArray& src); // deep copy
        ~NDArray(); 

        NDArray& cpu();
        NDArray& gpu();
        NDArray& to(Device device);

        NDArray& reshape(const shape_t shape);
        NDArray& T2d(); // transpose
        NDArray& swap(const int axis1, const int axis2); //! Only useable for NDArray on GPU

        bool operator == (const NDArray& B); // element-wise comparison
        NDArray& operator + (const NDArray& B); // element-wise addition
        NDArray& operator - (const NDArray& B); // element-wise subtraction
        NDArray& operator * (const NDArray& B); // 2D matrix multiplic
        NDArray& operator / (const NDArray& B); // element-wise division

        static int shape2size(const shape_t shape);
};

NDArray& mul(const NDArray& A, const NDArray& B); // element-wise multiplication
NDArray& mul_scalar(const NDArray& A, const scalar_t B); 

NDArray& add_scalar(const NDArray& A, const scalar_t B);
NDArray& sub_scalar(const NDArray& A, const scalar_t B);
NDArray& div_scalar(const NDArray& A, const scalar_t B);

NDArray& pow_scalar(const NDArray& A, const scalar_t B); 
NDArray& log_(const NDArray& A);
NDArray& exp_(const NDArray& A);

#endif
