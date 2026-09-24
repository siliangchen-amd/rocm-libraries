/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    typedef enum rocsparse_csrsort_alg_
    {
        rocsparse_csrsort_alg_default = 0
    } rocsparse_csrsort_alg;

    template <typename I, typename J>
    rocsparse_status csrsort_buffer_size_template(rocsparse_handle handle,
                                                  int64_t          m,
                                                  int64_t          n,
                                                  int64_t          nnz,
                                                  const void*      csr_row_ptr,
                                                  const void*      csr_col_ind,
                                                  size_t*          buffer_size);

    // Sorts the column indices within each row. If perm is not null, the sort applies its
    // reordering to perm.
    template <typename I, typename J>
    rocsparse_status csrsort_template(rocsparse_handle     handle,
                                      int64_t              m,
                                      int64_t              n,
                                      int64_t              nnz,
                                      rocsparse_index_base idx_base,
                                      const void*          csr_row_ptr,
                                      void*                csr_col_ind,
                                      void*                perm,
                                      void*                temp_buffer);

    rocsparse_status csrsort_buffer_size(rocsparse_handle      handle,
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
                                         size_t*               buffer_size);

    // Sorts the column indices and values within each row of the CSR matrix A into the CSR
    // matrix B, whose row pointer receives a copy of the row pointer of A. Each output array
    // may either alias its input array, in which case it is sorted in place, or not overlap
    // it at all. The index types, data types and index bases of A and B must be identical.
    // Each of the batch_count matrices is sorted independently, with the batch strides given
    // in number of elements.
    rocsparse_status csrsort(rocsparse_handle      handle,
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
                             void*                 temp_buffer);
}
