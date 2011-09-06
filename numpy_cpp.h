#ifndef _NUMPY_CPP_H_
#define _NUMPY_CPP_H_

// Include the main NumPy header
extern "C" {
#include <numpy/ndarrayobject.h>
}

#include <complex>

namespace numpy {

// Exception thrown when the Python error is set (PyErr_Occurred() is true).
// This should be caught and an error code (NULL or -1) should be
// returned to Python or NumPy as appropriate.
class pyerr_exception : public std::exception {
    public:
        const char *what() const throw() {
            return "python error has been set";
        }
};

class pyerr_typeexception : public pyerr_exception {
    public:
        const char *what() const throw() {
            return "python type error has been set";
        }
};

// Type traits for the NumPy types
template <typename T> struct type_num_of;

// This is dodgy - need sizeof(bool) == 1 consistently for this to be valid...
//template <> struct type_num_of<bool> {
//    enum {value = NPY_BOOL};
//};
template <> struct type_num_of<npy_byte> {
    enum {value = NPY_BYTE};
};
template <> struct type_num_of<npy_ubyte> {
    enum {value = NPY_UBYTE};
};
template <> struct type_num_of<npy_short> {
    enum {value = NPY_SHORT};
};
template <> struct type_num_of<npy_ushort> {
    enum {value = NPY_USHORT};
};
template <> struct type_num_of<npy_int> {
    enum {value = NPY_INT};
};
template <> struct type_num_of<npy_uint> {
    enum {value = NPY_UINT};
};
template <> struct type_num_of<npy_long> {
    enum {value = NPY_LONG};
};
template <> struct type_num_of<npy_ulong> {
    enum {value = NPY_ULONG};
};
template <> struct type_num_of<npy_longlong> {
    enum {value = NPY_LONGLONG};
};
template <> struct type_num_of<npy_ulonglong> {
    enum {value = NPY_ULONGLONG};
};
template <> struct type_num_of<npy_float> {
    enum {value = NPY_FLOAT};
};
template <> struct type_num_of<npy_double> {
    enum {value = NPY_DOUBLE};
};
template <> struct type_num_of<npy_longdouble> {
    enum {value = NPY_LONGDOUBLE};
};
template <> struct type_num_of<npy_cfloat> {
    enum {value = NPY_CFLOAT};
};
template <> struct type_num_of< std::complex<npy_float> > {
    enum {value = NPY_CFLOAT};
};
template <> struct type_num_of<npy_cdouble> {
    enum {value = NPY_CDOUBLE};
};
template <> struct type_num_of< std::complex<npy_double> > {
    enum {value = NPY_CDOUBLE};
};
template <> struct type_num_of<npy_clongdouble> {
    enum {value = NPY_CLONGDOUBLE};
};
template <> struct type_num_of< std::complex<npy_longdouble> > {
    enum {value = NPY_CLONGDOUBLE};
};
template <> struct type_num_of<PyObject *> {
    enum {value = NPY_OBJECT};
};
template <typename T> struct type_num_of<T&> {
    enum {value = type_num_of<T>::value};
};
template <typename T> struct type_num_of<const T> {
    enum {value = type_num_of<T>::value};
};

template <typename T> struct is_const {
    enum {value = false};
};
template <typename T> struct is_const<const T> {
    enum {value = true};
};

namespace detail {
    template <typename AV, typename T, int ND> class array_view_accessors;

    template <typename AV, typename T>
    class array_view_accessors<AV, T, 1> {
    public:
        T& operator()(npy_intp i) {
            AV *self = static_cast<AV *>(this);

            return self->m_data[self->m_strides[0]*i];
        }

        const T& operator()(npy_intp i) const {
            AV *self = static_cast<AV *>(this);

            return self->m_data[self->m_strides[0]*i];
        }
    };

    template <typename AV, typename T>
    class array_view_accessors<AV, T, 2> {
    public:
        T& operator()(npy_intp i, npy_intp j) {
            AV *self = static_cast<AV *>(this);

            return self->m_data[self->m_strides[0]*i + self->m_strides[1]*j];
        }

        const T& operator()(npy_intp i, npy_intp j) const {
            AV *self = static_cast<AV *>(this);

            return self->m_data[self->m_strides[0]*i + self->m_strides[1]*j];
        }
    };

    template <typename AV, typename T>
    class array_view_accessors<AV, T, 3> {
    public:
        T& operator()(npy_intp i, npy_intp j, npy_intp k) {
            AV *self = static_cast<AV *>(this);

            return self->m_data[self->m_strides[0]*i +
                                self->m_strides[1]*j +
                                self->m_strides[2]*k];
        }

        const T& operator()(npy_intp i, npy_intp j, npy_intp k) const {
            AV *self = static_cast<AV *>(this);

            return self->m_data[self->m_strides[0]*i +
                                self->m_strides[1]*j +
                                self->m_strides[2]*k];
        }
    };
}

// Provides a view into a PyArrayObject as requested. If the data type
// of 'arr' doesn't match, will create a temporary copy with NPY_COPY
// or NPY_UPDATEIFCOPY as appropriate.
//
// Read only 2D view, only allowing 'safe' casting to create the view:
// array_view<const double, 2> view(arr, NPY_SAFE_CASTING);
//
// Read-write 3D view, allowing 'same_kind' casting to create the view:
// array_view<double, 3> view(arr, NPY_SAME_KIND_CASTING);
//
// TODO: This is very much WIP, class is not tested yet
template<typename T, int ND>
class array_view :
        public detail::array_view_accessors<array_view<T,ND>, T, ND> {
protected:
    PyArrayObject *m_arr;
    // Copies of the array data
    npy_intp m_shape[ND], m_strides[ND];
    T *m_data;
public:
    typedef T value_type;
    enum {ndim = ND};
    // arr:   The NumPy array object.
    // flags: Controls what casting may occur when making a copy to
    //        create the view.
    array_view(PyObject *arr, NPY_CASTING casting) {
        PyArrayObject *tmp;

        /* First make sure it's an array */
        if (casting == NPY_NO_CASTING || casting == NPY_EQUIV_CASTING) {
            if (PyArray_Check(arr)) {
                if (PyArray_NDIM(arr) != ND) {
                    PyErr_SetString(PyExc_TypeError,
                            "array has the wrong number of dimensions");
                    throw pyerr_typeexception();
                }
                tmp = (PyArrayObject *)arr;
                Py_INCREF(tmp);
            }
            else {
                PyErr_SetString(PyExc_TypeError,
                        "cannot cast object to requested type");
                throw pyerr_typeexception();
            }
        }
        else {
            tmp = (PyArrayObject *)PyArray_FromAny(arr, NULL, ND, ND,
                                is_const<T>::value ? 0 : NPY_WRITEABLE, NULL);
            if (tmp == NULL) {
                throw pyerr_exception();
            }
        }

        /* Check if no copy is needed */
        if (PyArray_ISALIGNED(tmp) &&
                PyArray_ISNBO(PyArray_DESCR(tmp)->byteorder) &&
                PyArray_DESCR(tmp)->type_num == type_num_of<T>::value) {
            m_arr = tmp;
        }
        /* Check the casting rules, and copy */
        else {
            PyArray_Descr *dtype = PyArray_DescrFromType(type_num_of<T>::value);
            if (dtype == NULL) {
                Py_DECREF(tmp);
                throw pyerr_exception();
            }
            if (!PyArray_CanCastTypeTo(PyArray_DESCR(tmp), dtype, casting)) {
                Py_DECREF(dtype);
                Py_DECREF(tmp);
                PyErr_SetString(PyExc_TypeError,
                        "cannot cast array to requested type");
                throw pyerr_typeexception();
            }
            if (!is_const<T>::value &&
                        !PyArray_CanCastTypeTo(dtype, PyArray_DESCR(tmp),
                                                        casting)) {
                Py_DECREF(dtype);
                Py_DECREF(tmp);
                PyErr_SetString(PyExc_TypeError,
                        "cannot cast requested type back to writeable array");
                throw pyerr_typeexception();
            }

            m_arr = (PyArrayObject *)PyArray_FromArray(tmp, dtype, NPY_ALIGNED|
                                    is_const<T>::value ? 0 : NPY_WRITEABLE);
            Py_DECREF(tmp);
            if (m_arr == NULL) {
                throw pyerr_exception();
            }
        }

        /* Copy some of the data to the view object for faster access */
        memcpy(m_shape, PyArray_DIMS(m_arr), ND*sizeof(npy_intp));
        memcpy(m_strides, PyArray_STRIDES(m_arr), ND*sizeof(npy_intp));
        for (int i = 0; i < ND; ++i) {
            m_strides[i] /= sizeof(T);
        }
    }

    ~array_view() {
        Py_DECREF(m_arr);
    }
};

class raw_array_helper {
    // Data for once a layout is set
    int m_type_num;
    char *m_data;
    bool m_writeable;

    // Data for while try_* is being called.  Once a layout is set,
    // m_arr owns a reference to the input object array or an array
    // copy created to make the layout fit.
    PyObject *m_obj;
    PyArrayObject *m_arr;
    PyArray_Descr *m_dtype;
    int m_ndim;
    npy_intp m_dims[NPY_MAXDIMS];

    // No copying, no default constructor
    raw_array_helper(const raw_array_helper&);
    raw_array_helper();
    raw_array_helper& operator=(const raw_array_helper&);
public:
    // Initialize the raw array helper for viewing object 'arr'.
    // Set 'writeable' to true if you want to be able to write
    // to the array and have it affect 'arr'.
    raw_array_helper(PyObject *obj, bool writeable = false)
        : m_writeable(writeable), m_type_num(NPY_NTYPES), m_data(NULL),
            m_dtype(NULL), m_arr(NULL), m_obj(NULL)
    {
        if (PyArray_GetArrayParamsFromObject(obj, NULL, writeable,
                            &m_dtype, &m_ndim, m_dims, &m_arr, NULL) < 0) {
            throw pyerr_exception();
        }
        // Copy the dimensions to simplify the try_* functions
        if (m_arr) {
            m_ndim = PyArray_NDIM(m_arr);
            memcpy(m_dims, PyArray_DIMS(m_arr), m_ndim*sizeof(npy_intp));
        }
        else {
            Py_INCREF(obj);
            m_obj = obj;
        }
    }

    ~raw_array_helper()
    {
        Py_XDECREF(m_obj);
        Py_XDECREF(m_arr);
        Py_XDECREF(m_dtype);
    }

    // Tries to set the raw array data based on the specifications.
    //
    // (dim0) allows a specific dimension to be used, pass the value
    // 0 to indicate no constraint.
    //
    // The parameter 'casting' indicates what data conversions are
    // to be permitted. A typical idiom is to first try all the
    // specialized implementations with NPY_EQUIV_CASTING, then end
    // with a default type such as NPY_SAME_KIND_CASTING.
    template<typename T>
    bool try_1d(int dim0, NPY_CASTING casting)
    {
        if (m_data) {
            PyErr_SetString(PyExc_RuntimeError,
                            "called try_1d after another try_* succeeded "
                            "in raw_array_helper");
            throw pyerr_exception();
        }

        // Validate the dimensions
        if (m_ndim != 1 || (dim0 != 0 && m_dims[0] != dim0)) {
            return false;
        }

        if (m_arr) {
            // Check if the type matches exactly and the data is contiguous
            if (PyArray_IS_C_CONTIGUOUS(m_arr) &&
                            PyArray_EquivTypenums(type_num_of<T>::value,
                                          PyArray_DESCR(m_arr)->type_num) &&
                            PyArray_ISNBO(PyArray_DESCR(m_arr)->byteorder)) {
                m_data = PyArray_BYTES(m_arr);
                m_type_num = type_num_of<T>::value;
                return true;
            }
            // Check if the type can be cast according to 'casting'
            else {
                PyArray_Descr *dtype = PyArray_DescrFromType(
                                                type_num_of<T>::value);
                if (PyArray_CanCastArrayTo(m_arr, dtype, casting)) {
                    PyArrayObject *temp = PyArray_FromArray(
                                            m_arr, dtype,
                                            NPY_FORCECAST|
                                        (m_writeable ? NPY_UPDATEIFCOPY : 0)|
                                            NPY_C_CONTIGUOUS);
                    if (temp == NULL) {
                        throw pyerr_exception();
                    }
                    Py_DECREF(m_arr);
                    m_arr = temp;

                    m_data = PyArray_BYTES(m_arr);
                    m_type_num = type_num_of<T>::value;
                    return true;
                }
                else {
                    Py_DECREF(dtype);
                }
            }
        }
        else {
            PyArray_Descr *dtype = PyArray_DescrFromType(
                                            type_num_of<T>::value);
            if (PyArray_CanCastTypeTo(m_dtype, dtype, casting)) {
                m_arr = PyArray_NewFromDescr(&PyArray_Type, dtype,
                                            1, m_dims, NULL, NULL,
                                            0, NULL);
                if (m_arr == NULL) {
                    throw pyerr_exception();
                }
                if (PyArray_CopyObject(m_arr, m_obj) < 0) {
                    throw pyerr_exception();
                }
                Py_DECREF(m_obj);
                m_obj = NULL;
                Py_DECREF(m_dtype);
                m_dtype = NULL;

                m_data = PyArray_BYTES(m_arr);
                m_type_num = type_num_of<T>::value;
                return true;
            }
            else {
                Py_DECREF(dtype);
            }
        }

        return false;
    }

    // Tries to set the raw array data based on the specifications.
    //
    // (dim0, dim1) allow specific dimensions to be specified exactly,
    // with value 0 indicating no constraint.  For example, (0, 3)
    // is to request a 1D array of 3-vectors.
    //
    // (axis0, axis1) is a permutation indicating the axis ordering,
    // with 0 meaning the smallest axis stride and ndim-1 meaning the
    // largest axis stride.
    // (1, 0) = C contiguous, (0, 1) = F contiguous.
    //
    // The flag allow_axis_conversion indicates whether to return false rather
    // than make a copy of the array to satisfy the axis ordering.
    // This is for use when you have both C and F routines, and want
    // to get the right one.
    //
    // The parameter 'casting' indicates what data conversions are
    // to be permitted. A typical idiom is to first try all the
    // specialized implementations with NPY_EQUIV_CASTING, then end
    // with a default type such as NPY_SAME_KIND_CASTING.
    template <typename T>
    bool try_2d(int dim0, int dim1, int axis0, int axis1,
                bool allow_axis_conversion, NPY_CASTING casting)
    {
        if (m_data) {
            PyErr_SetString(PyExc_RuntimeError,
                            "called try_2d after another try_* succeeded "
                            "in raw_array_helper");
            throw pyerr_exception();
        }

        // Validate the dimensions
        if (m_ndim != 2 ||
                    (dim0 != 0 && m_dims[0] != dim0) ||
                    (dim1 != 0 && m_dims[1] != dim1)) {
            return false;
        }
        if (!((axis0 == 0 && axis1 == 1) || (axis0 == 1 && axis1 == 0))) {
            PyErr_SetString(PyExc_RuntimeError,
                        "invalid dims ordering specified for try_2d in "
                        "raw_array_helper");
            throw pyerr_exception();
        }

        if (m_arr) {
            bool needs_axis_conversion = true;
            if (axis0 == 0) {
                if (PyArray_IS_F_CONTIGUOUS(m_arr)) {
                    needs_axis_conversion = false;
                }
            }
            else if (axis0 == 1) {
                if (PyArray_IS_C_CONTIGUOUS(m_arr)) {
                    needs_axis_conversion = false;
                }
            }
            // Check if the type matches exactly and the data is contiguous
            if (!needs_axis_conversion &&
                            PyArray_EquivTypenums(type_num_of<T>::value,
                                          PyArray_DESCR(m_arr)->type_num) &&
                            PyArray_ISNBO(PyArray_DESCR(m_arr)->byteorder)) {
                m_data = PyArray_BYTES(m_arr);
                m_type_num = type_num_of<T>::value;
                return true;
            }
            // Check if the type can be cast according to 'casting'
            else if (!needs_axis_conversion || allow_axis_conversion) {
                PyArray_Descr *dtype = PyArray_DescrFromType(
                                                type_num_of<T>::value);
                if (PyArray_CanCastArrayTo(m_arr, dtype, casting)) {
                    PyArrayObject *temp = PyArray_FromArray(
                                            m_arr, dtype,
                                            NPY_FORCECAST|
                                        (m_writeable ? NPY_UPDATEIFCOPY : 0)|
                          (axis0 == 0 ? NPY_F_CONTIGUOUS : NPY_C_CONTIGUOUS));
                    if (temp == NULL) {
                        throw pyerr_exception();
                    }
                    Py_DECREF(m_arr);
                    m_arr = temp;

                    m_data = PyArray_BYTES(m_arr);
                    m_type_num = type_num_of<T>::value;
                    return true;
                }
                else {
                    Py_DECREF(dtype);
                }
            }
        }
        else {
            PyArray_Descr *dtype = PyArray_DescrFromType(
                                            type_num_of<T>::value);
            if (PyArray_CanCastTypeTo(m_dtype, dtype, casting)) {
                m_arr = PyArray_NewFromDescr(&PyArray_Type, dtype,
                                            2, m_dims, NULL, NULL,
                                          (axis0 == 0 ? NPY_F_CONTIGUOUS : 0),
                                            NULL);
                if (m_arr == NULL) {
                    throw pyerr_exception();
                }
                if (PyArray_CopyObject(m_arr, m_obj) < 0) {
                    throw pyerr_exception();
                }
                Py_DECREF(m_obj);
                m_obj = NULL;
                Py_DECREF(m_dtype);
                m_dtype = NULL;

                m_data = PyArray_BYTES(m_arr);
                m_type_num = type_num_of<T>::value;
                return true;
            }
            else {
                Py_DECREF(dtype);
            }
        }

        return false;
    }
        
};

} // namespace numpy

#endif
