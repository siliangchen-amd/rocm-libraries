/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "internal/conversion/rocsparse_csrsort.h"
#include "rocsparse_utility.hpp"

#include "../level1/rocsparse_gthr.hpp"
#include "csrsort_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsort.hpp"
#include "rocsparse_gcreate_identity_permutation.hpp"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    // Number of bits needed to represent the column indices, which are at most n.
    static uint32_t csrsort_endbit(int64_t n)
    {
        // __builtin_clzll is undefined for n == 0
        return (n == 0) ? 0 : 64 - __builtin_clzll(static_cast<unsigned long long>(n));
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrsort_buffer_size_template(rocsparse_handle handle,
                                                         int64_t          m,
                                                         int64_t          n,
                                                         int64_t          nnz,
                                                         const void*      csr_row_ptr,
                                                         const void*      csr_col_ind,
                                                         size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0 || nnz == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    const uint32_t startbit = 0;
    const uint32_t endbit   = rocsparse::csrsort_endbit(n);

    // We do not know if sort_pairs or sort_keys will be called, so use the largest buffer between the two
    size_t size1;
    size_t size2;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, I, I>(
            handle, nnz, m, startbit, endbit, &size1)));
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
        handle, nnz, m, startbit, endbit, &size2)));

    *buffer_size = rocsparse::align_size<char>(rocsparse::max(size1, size2));

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays

    // columns buffer
    *buffer_size += rocsparse::align_size<J>(nnz);
    // perm buffer
    *buffer_size += rocsparse::align_size<I>(nnz);
    // segm buffer
    *buffer_size += rocsparse::align_size<I>(m + 1);

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csrsort_template(rocsparse_handle     handle,
                                             int64_t              m,
                                             int64_t              n,
                                             int64_t              nnz,
                                             rocsparse_index_base idx_base,
                                             const void*          csr_row_ptr,
                                             void*                csr_col_ind,
                                             void*                perm,
                                             void*                temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    const I* csr_row_ptr_ = reinterpret_cast<const I*>(csr_row_ptr);
    J*       csr_col_ind_ = reinterpret_cast<J*>(csr_col_ind);
    I*       perm_        = reinterpret_cast<I*>(perm);

    // Stream
    hipStream_t stream = handle->stream;

    const uint32_t startbit = 0;
    const uint32_t endbit   = rocsparse::csrsort_endbit(n);
    size_t         size;

    if(perm_ != nullptr)
    {
        // Sort pairs, if permutation vector is present
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, I, I>(
                handle, nnz, m, startbit, endbit, &size)));
    }
    else
    {
        // Sort keys, if no permutation vector is present
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
                handle, nnz, m, startbit, endbit, &size)));
    }

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // columns buffer
    J* tmp_cols = reinterpret_cast<J*>(ptr);
    ptr += rocsparse::align_size<J>(nnz);

    // perm buffer
    I* tmp_perm = reinterpret_cast<I*>(ptr);
    ptr += rocsparse::align_size<I>(nnz);

    // segm buffer
    I* tmp_segm = reinterpret_cast<I*>(ptr);
    ptr += rocsparse::align_size<I>(m + 1);

    // Index base one requires shift of offset positions
    if(idx_base == rocsparse_index_base_one)
    {
#define CSRSORT_DIM 512
        dim3 csrsort_blocks((m + 1 - 1) / CSRSORT_DIM + 1);
        dim3 csrsort_threads(CSRSORT_DIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrsort_shift_kernel<CSRSORT_DIM>),
                                           csrsort_blocks,
                                           csrsort_threads,
                                           0,
                                           stream,
                                           m + 1,
                                           csr_row_ptr_,
                                           tmp_segm);
#undef CSRSORT_DIM
    }

    // rocprim buffer
    void* tmp_rocprim = reinterpret_cast<void*>(ptr);

    // Switch between offsets
    const I* offsets = (idx_base == rocsparse_index_base_one) ? tmp_segm : csr_row_ptr_;

    // Sort by columns and obtain permutation vector

    if(perm_ != nullptr)
    {
        // Sort by pairs, if permutation vector is present
        rocsparse::primitives::double_buffer<J> keys(csr_col_ind_, tmp_cols);
        rocsparse::primitives::double_buffer<I> vals(perm_, tmp_perm);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_pairs(
            handle, keys, vals, nnz, m, offsets, offsets + 1, startbit, endbit, size, tmp_rocprim));

        if(keys.current() != csr_col_ind_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(csr_col_ind_,
                                                         keys.current(),
                                                         sizeof(J) * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         stream));
        }
        if(vals.current() != perm_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                perm_, vals.current(), sizeof(I) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }
    else
    {
        // Sort by keys, if no permutation vector is present
        rocsparse::primitives::double_buffer<J> keys(csr_col_ind_, tmp_cols);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_keys(
            handle, keys, nnz, m, offsets, offsets + 1, startbit, endbit, size, tmp_rocprim));

        if(keys.current() != csr_col_ind_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(csr_col_ind_,
                                                         keys.current(),
                                                         sizeof(J) * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         stream));
        }
    }
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_csrsort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csrsort_buffer_size",
                         m,
                         n,
                         nnz,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsort_buffer_size_template<rocsparse_int, rocsparse_int>(
        handle, m, n, nnz, csr_row_ptr, csr_col_ind, buffer_size)));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csrsort(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const rocsparse_int*      csr_row_ptr,
                                              rocsparse_int*            csr_col_ind,
                                              rocsparse_int*            perm,
                                              void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csrsort",
                         m,
                         n,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, temp_buffer);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsort_template<rocsparse_int, rocsparse_int>(
        handle, m, n, nnz, descr->base, csr_row_ptr, csr_col_ind, perm, temp_buffer)));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#define INSTANTIATE(I, J)                                                                          \
    template rocsparse_status rocsparse::csrsort_buffer_size_template<I, J>(                       \
        rocsparse_handle handle,                                                                   \
        int64_t          m,                                                                        \
        int64_t          n,                                                                        \
        int64_t          nnz,                                                                      \
        const void*      csr_row_ptr,                                                              \
        const void*      csr_col_ind,                                                              \
        size_t*          buffer_size);                                                             \
    template rocsparse_status rocsparse::csrsort_template<I, J>(rocsparse_handle     handle,       \
                                                                int64_t              m,            \
                                                                int64_t              n,            \
                                                                int64_t              nnz,          \
                                                                rocsparse_index_base idx_base,     \
                                                                const void*          csr_row_ptr,  \
                                                                void*                csr_col_ind,  \
                                                                void*                perm,         \
                                                                void*                temp_buffer)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

namespace rocsparse
{
    // The buffer starts with the permutation array, which tracks where each entry moves
    // while the column indices are sorted, followed by scratch space shared by the index
    // sort and the value permutation.
    static size_t csrsort_perm_size(int64_t nnz, rocsparse_indextype perm_indextype)
    {
        return rocsparse::align_size<char>(rocsparse::indextype_sizeof(perm_indextype) * nnz);
    }

    // The row pointer and column indices are copied and the values gathered without any
    // type conversion.
    static rocsparse_status csrsort_check_types(rocsparse_indextype csr_row_ptr_indextype_A,
                                                rocsparse_indextype csr_col_indextype_A,
                                                rocsparse_datatype  csr_val_datatype_A,
                                                rocsparse_indextype csr_row_ptr_indextype_B,
                                                rocsparse_indextype csr_col_indextype_B,
                                                rocsparse_datatype  csr_val_datatype_B)
    {
        if(csr_row_ptr_indextype_B != csr_row_ptr_indextype_A
           || csr_col_indextype_B != csr_col_indextype_A)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the index types of the output matrix must match the index types of the input "
                "matrix");
        }
        if(csr_val_datatype_B != csr_val_datatype_A)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the data type of the output matrix must match the data type of the input matrix");
        }
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::csrsort_buffer_size(rocsparse_handle      handle,
                                                rocsparse_csrsort_alg alg,
                                                int64_t               m,
                                                int64_t               n,
                                                int64_t               nnz,
                                                rocsparse_indextype   csr_row_ptr_indextype_A,
                                                const void*           csr_row_ptr_A,
                                                rocsparse_indextype   csr_col_indextype_A,
                                                const void*           csr_col_ind_A,
                                                rocsparse_datatype    csr_val_datatype_A,
                                                const void*           csr_val_A,
                                                rocsparse_indextype   csr_row_ptr_indextype_B,
                                                const void*           csr_row_ptr_B,
                                                rocsparse_indextype   csr_col_indextype_B,
                                                const void*           csr_col_ind_B,
                                                rocsparse_datatype    csr_val_datatype_B,
                                                const void*           csr_val_B,
                                                size_t*               buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_check_types(csr_row_ptr_indextype_A,
                                                             csr_col_indextype_A,
                                                             csr_val_datatype_A,
                                                             csr_row_ptr_indextype_B,
                                                             csr_col_indextype_B,
                                                             csr_val_datatype_B));

    auto f = rocsparse::csrsort_buffer_size_template<int32_t, int32_t>;
    if(csr_row_ptr_indextype_B == rocsparse_indextype_i32
       && csr_col_indextype_B == rocsparse_indextype_i32)
    {
        f = rocsparse::csrsort_buffer_size_template<int32_t, int32_t>;
    }
    else if(csr_row_ptr_indextype_B == rocsparse_indextype_i64
            && csr_col_indextype_B == rocsparse_indextype_i32)
    {
        f = rocsparse::csrsort_buffer_size_template<int64_t, int32_t>;
    }
    else if(csr_row_ptr_indextype_B == rocsparse_indextype_i64
            && csr_col_indextype_B == rocsparse_indextype_i64)
    {
        f = rocsparse::csrsort_buffer_size_template<int64_t, int64_t>;
    }
    else
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "rocsparse::csrsort_buffer_size failed from dispatching");
    }

    size_t sort_buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        f(handle, m, n, nnz, csr_row_ptr_B, csr_col_ind_B, &sort_buffer_size));

    // Values sorted in place are gathered into scratch space first, since the gather cannot
    // write over its own input.
    const size_t gather_buffer_size
        = (csr_val_B == csr_val_A)
              ? rocsparse::align_size<char>(rocsparse::datatype_sizeof(csr_val_datatype_B) * nnz)
              : 0;

    *buffer_size = rocsparse::csrsort_perm_size(nnz, csr_row_ptr_indextype_B)
                   + rocsparse::max(sort_buffer_size, gather_buffer_size);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsort(rocsparse_handle      handle,
                                    rocsparse_csrsort_alg alg,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               nnz,
                                    int64_t               batch_count_A,
                                    int64_t               offsets_batch_stride_A,
                                    int64_t               columns_values_batch_stride_A,
                                    rocsparse_index_base  idx_base_A,
                                    rocsparse_indextype   csr_row_ptr_indextype_A,
                                    const void*           csr_row_ptr_A,
                                    rocsparse_indextype   csr_col_indextype_A,
                                    const void*           csr_col_ind_A,
                                    rocsparse_datatype    csr_val_datatype_A,
                                    const void*           csr_val_A,
                                    int64_t               batch_count_B,
                                    int64_t               offsets_batch_stride_B,
                                    int64_t               columns_values_batch_stride_B,
                                    rocsparse_index_base  idx_base_B,
                                    rocsparse_indextype   csr_row_ptr_indextype_B,
                                    void*                 csr_row_ptr_B,
                                    rocsparse_indextype   csr_col_indextype_B,
                                    void*                 csr_col_ind_B,
                                    rocsparse_datatype    csr_val_datatype_B,
                                    void*                 csr_val_B,
                                    void*                 temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_check_types(csr_row_ptr_indextype_A,
                                                             csr_col_indextype_A,
                                                             csr_val_datatype_A,
                                                             csr_row_ptr_indextype_B,
                                                             csr_col_indextype_B,
                                                             csr_val_datatype_B));

    if(batch_count_B != batch_count_A)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the batch count of the output matrix must match the batch count of the input matrix");
    }

    if(idx_base_B != idx_base_A)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the index base of the output matrix must match the index base of the input matrix");
    }

    auto f = rocsparse::csrsort_template<int32_t, int32_t>;
    if(csr_row_ptr_indextype_B == rocsparse_indextype_i32
       && csr_col_indextype_B == rocsparse_indextype_i32)
    {
        f = rocsparse::csrsort_template<int32_t, int32_t>;
    }
    else if(csr_row_ptr_indextype_B == rocsparse_indextype_i64
            && csr_col_indextype_B == rocsparse_indextype_i32)
    {
        f = rocsparse::csrsort_template<int64_t, int32_t>;
    }
    else if(csr_row_ptr_indextype_B == rocsparse_indextype_i64
            && csr_col_indextype_B == rocsparse_indextype_i64)
    {
        f = rocsparse::csrsort_template<int64_t, int64_t>;
    }
    else
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "rocsparse::csrsort failed from dispatching");
    }

    const rocsparse_indextype perm_indextype = csr_row_ptr_indextype_B;

    const size_t row_ptr_size_A = rocsparse::indextype_sizeof(csr_row_ptr_indextype_A);
    const size_t col_size_A     = rocsparse::indextype_sizeof(csr_col_indextype_A);
    const size_t val_size_A     = rocsparse::datatype_sizeof(csr_val_datatype_A);
    const size_t row_ptr_size_B = rocsparse::indextype_sizeof(csr_row_ptr_indextype_B);
    const size_t col_size_B     = rocsparse::indextype_sizeof(csr_col_indextype_B);
    const size_t val_size_B     = rocsparse::datatype_sizeof(csr_val_datatype_B);

    void* perm = temp_buffer;
    void* sort_buffer
        = reinterpret_cast<char*>(temp_buffer) + rocsparse::csrsort_perm_size(nnz, perm_indextype);

    // The batches run one after the other on the handle stream, so they share temp_buffer.
    for(int64_t batch = 0; batch < batch_count_A; ++batch)
    {
        const int64_t offsets_offset_A = batch * offsets_batch_stride_A;
        const int64_t offsets_offset_B = batch * offsets_batch_stride_B;
        const int64_t cv_offset_A      = batch * columns_values_batch_stride_A;
        const int64_t cv_offset_B      = batch * columns_values_batch_stride_B;

        const void* row_ptr_A
            = reinterpret_cast<const char*>(csr_row_ptr_A) + offsets_offset_A * row_ptr_size_A;
        const void* col_ind_A
            = reinterpret_cast<const char*>(csr_col_ind_A) + cv_offset_A * col_size_A;
        const void* val_A = reinterpret_cast<const char*>(csr_val_A) + cv_offset_A * val_size_A;
        void*       row_ptr_B
            = reinterpret_cast<char*>(csr_row_ptr_B) + offsets_offset_B * row_ptr_size_B;
        void* col_ind_B = reinterpret_cast<char*>(csr_col_ind_B) + cv_offset_B * col_size_B;
        void* val_B     = reinterpret_cast<char*>(csr_val_B) + cv_offset_B * val_size_B;

        // The column sort works in place, so the row pointer and column indices of A are
        // first copied into B. A matrix without rows may have a null row pointer.
        if(m > 0 && row_ptr_B != row_ptr_A)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(row_ptr_B,
                                               row_ptr_A,
                                               row_ptr_size_A * (m + 1),
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }
        if(col_ind_B != col_ind_A)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(col_ind_B,
                                               col_ind_A,
                                               col_size_A * nnz,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }

        // The column sort applies its reordering to perm, so it must start as the identity.
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gcreate_identity_permutation(handle, nnz, perm_indextype, perm));

        RETURN_IF_ROCSPARSE_ERROR(
            f(handle, m, n, nnz, idx_base_B, row_ptr_B, col_ind_B, perm, sort_buffer));

        // The gather cannot write over its own input, so in place values go through scratch.
        const bool in_place_val = (val_B == val_A);
        void*      sorted_val   = in_place_val ? sort_buffer : val_B;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthr(handle,
                                                  nnz,
                                                  csr_val_datatype_A,
                                                  val_A,
                                                  csr_val_datatype_B,
                                                  sorted_val,
                                                  perm_indextype,
                                                  perm,
                                                  rocsparse_index_base_zero));

        if(in_place_val)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(val_B,
                                               sorted_val,
                                               val_size_B * nnz,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }
    }

    return rocsparse_status_success;
}
