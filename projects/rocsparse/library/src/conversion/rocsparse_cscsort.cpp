/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_cscsort.h"
#include "rocsparse_utility.hpp"

#include "rocsparse_control.hpp"
#include "rocsparse_cscsort.hpp"
#include "rocsparse_csrsort.hpp"

template <typename I, typename J>
rocsparse_status rocsparse::cscsort_buffer_size_template(rocsparse_handle handle,
                                                         int64_t          m,
                                                         int64_t          n,
                                                         int64_t          nnz,
                                                         const void*      csc_col_ptr,
                                                         const void*      csc_row_ind,
                                                         size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Sorting the row indices within each column of A is sorting the column indices within
    // each row of the transpose of A, stored in CSR format.
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsort_buffer_size_template<I, J>(
        handle, n, m, nnz, csc_col_ptr, csc_row_ind, buffer_size)));
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::cscsort_template(rocsparse_handle     handle,
                                             int64_t              m,
                                             int64_t              n,
                                             int64_t              nnz,
                                             rocsparse_index_base idx_base,
                                             const void*          csc_col_ptr,
                                             void*                csc_row_ind,
                                             void*                perm,
                                             void*                temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsort_template<I, J>(
        handle, n, m, nnz, idx_base, csc_col_ptr, csc_row_ind, perm, temp_buffer)));
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_cscsort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csc_col_ptr,
                                                          const rocsparse_int* csc_row_ind,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, n, csc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::cscsort_buffer_size_template<rocsparse_int, rocsparse_int>(
        handle, m, n, nnz, csc_col_ptr, csc_row_ind, buffer_size)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_cscsort(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const rocsparse_int*      csc_col_ptr,
                                              rocsparse_int*            csc_row_ind,
                                              rocsparse_int*            perm,
                                              void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_ARRAY(5, n, csc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, temp_buffer);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::cscsort_template<rocsparse_int, rocsparse_int>(
        handle, m, n, nnz, descr->base, csc_col_ptr, csc_row_ind, perm, temp_buffer)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#define INSTANTIATE(I, J)                                                                          \
    template rocsparse_status rocsparse::cscsort_buffer_size_template<I, J>(                       \
        rocsparse_handle handle,                                                                   \
        int64_t          m,                                                                        \
        int64_t          n,                                                                        \
        int64_t          nnz,                                                                      \
        const void*      csc_col_ptr,                                                              \
        const void*      csc_row_ind,                                                              \
        size_t*          buffer_size);                                                             \
    template rocsparse_status rocsparse::cscsort_template<I, J>(rocsparse_handle     handle,       \
                                                                int64_t              m,            \
                                                                int64_t              n,            \
                                                                int64_t              nnz,          \
                                                                rocsparse_index_base idx_base,     \
                                                                const void*          csc_col_ptr,  \
                                                                void*                csc_row_ind,  \
                                                                void*                perm,         \
                                                                void*                temp_buffer)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

// Sorting the row indices within each column of A is sorting the column indices within each
// row of the transpose of A, stored in CSR format.

rocsparse_status rocsparse::cscsort_buffer_size(rocsparse_handle      handle,
                                                rocsparse_cscsort_alg alg,
                                                int64_t               m,
                                                int64_t               n,
                                                int64_t               nnz,
                                                rocsparse_indextype   csc_col_ptr_indextype_A,
                                                const void*           csc_col_ptr_A,
                                                rocsparse_indextype   csc_row_indextype_A,
                                                const void*           csc_row_ind_A,
                                                rocsparse_datatype    csc_val_datatype_A,
                                                const void*           csc_val_A,
                                                rocsparse_indextype   csc_col_ptr_indextype_B,
                                                const void*           csc_col_ptr_B,
                                                rocsparse_indextype   csc_row_indextype_B,
                                                const void*           csc_row_ind_B,
                                                rocsparse_datatype    csc_val_datatype_B,
                                                const void*           csc_val_B,
                                                size_t*               buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_buffer_size(handle,
                                                             rocsparse::rocsparse_csrsort_alg_default,
                                                             n,
                                                             m,
                                                             nnz,
                                                             csc_col_ptr_indextype_A,
                                                             csc_col_ptr_A,
                                                             csc_row_indextype_A,
                                                             csc_row_ind_A,
                                                             csc_val_datatype_A,
                                                             csc_val_A,
                                                             csc_col_ptr_indextype_B,
                                                             csc_col_ptr_B,
                                                             csc_row_indextype_B,
                                                             csc_row_ind_B,
                                                             csc_val_datatype_B,
                                                             csc_val_B,
                                                             buffer_size));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsort(rocsparse_handle      handle,
                                    rocsparse_cscsort_alg alg,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               nnz,
                                    int64_t               batch_count_A,
                                    int64_t               offsets_batch_stride_A,
                                    int64_t               rows_values_batch_stride_A,
                                    rocsparse_index_base  idx_base_A,
                                    rocsparse_indextype   csc_col_ptr_indextype_A,
                                    const void*           csc_col_ptr_A,
                                    rocsparse_indextype   csc_row_indextype_A,
                                    const void*           csc_row_ind_A,
                                    rocsparse_datatype    csc_val_datatype_A,
                                    const void*           csc_val_A,
                                    int64_t               batch_count_B,
                                    int64_t               offsets_batch_stride_B,
                                    int64_t               rows_values_batch_stride_B,
                                    rocsparse_index_base  idx_base_B,
                                    rocsparse_indextype   csc_col_ptr_indextype_B,
                                    void*                 csc_col_ptr_B,
                                    rocsparse_indextype   csc_row_indextype_B,
                                    void*                 csc_row_ind_B,
                                    rocsparse_datatype    csc_val_datatype_B,
                                    void*                 csc_val_B,
                                    void*                 temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort(handle,
                                                 rocsparse::rocsparse_csrsort_alg_default,
                                                 n,
                                                 m,
                                                 nnz,
                                                 batch_count_A,
                                                 offsets_batch_stride_A,
                                                 rows_values_batch_stride_A,
                                                 idx_base_A,
                                                 csc_col_ptr_indextype_A,
                                                 csc_col_ptr_A,
                                                 csc_row_indextype_A,
                                                 csc_row_ind_A,
                                                 csc_val_datatype_A,
                                                 csc_val_A,
                                                 batch_count_B,
                                                 offsets_batch_stride_B,
                                                 rows_values_batch_stride_B,
                                                 idx_base_B,
                                                 csc_col_ptr_indextype_B,
                                                 csc_col_ptr_B,
                                                 csc_row_indextype_B,
                                                 csc_row_ind_B,
                                                 csc_val_datatype_B,
                                                 csc_val_B,
                                                 temp_buffer));
    return rocsparse_status_success;
}
