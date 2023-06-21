import torch
from torch_sparse import SparseTensor, fill_diag, mul, spmm_sum_without_backward, matmul
from torch_sparse import sum as sparsesum
from torch_scatter import segment_csr


def distribute_work_for_row(rowptr: torch.Tensor, value: torch.Tensor, num_threads_on_row: int) -> tuple:
    flops_per_row = segment_csr(value, rowptr, None, "sum")
    num_rows = flops_per_row.shape[0]
    total_flops_on_rows = flops_per_row.sum()
    target_flops_on_rows = total_flops_on_rows // num_threads_on_row
    row_splits = torch.full((num_threads_on_row + 1,), num_rows, dtype=torch.int32)
    row_splits[0] = 0

    # use greedy algorithm to distribute work for row_splits
    cur_flops_sum = 0
    cur_tid = 0
    for i in range(flops_per_row.shape[0]):
        cur_flops_sum += flops_per_row[i]
        if cur_flops_sum > target_flops_on_rows and cur_tid < num_threads_on_row:
            row_splits[cur_tid+1] = i
            cur_tid += 1
            cur_flops_sum = flops_per_row[i]
    row_splits[-1] = num_rows

    print(row_splits)
    return row_splits

def disribute_work_for_col(other: torch.Tensor,  num_threads_on_col: int):
    num_cols = other.shape[-1]
    total_flops_on_cols = num_cols
    target_flops_on_cols = total_flops_on_cols // num_threads_on_col
    col_splits = torch.full((num_threads_on_col + 1,), num_cols, dtype=torch.int32)
    col_splits[0] = 0

    # to distributed work evenly for col_splits
    for i in range(num_threads_on_col):
        col_splits[i] = i * target_flops_on_cols
    col_splits[-1] = num_cols

    print(col_splits)
    return col_splits
    
def SPMM_forward(src: torch.Tensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row_splits, col_splits = src.get_work_range()
    # if row_splits is None
    if row_splits.size(0) == 0:
        num_threads = torch.get_num_threads()
        row_splits = distribute_work_for_row(rowptr, value, num_threads)

    # the other matrix might change, so we need to re-distribute work for col_splits every time
    col_splits = disribute_work_for_col(other, 1)

    src.set_work_range(row_splits, col_splits)
    
    if value is not None:
        value = value.to(other.dtype)
    return spmm_sum_without_backward(rowptr, col, value, other, out, row_splits, col_splits)

def SPMM_backward(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # rowptr, col, value = src.csr()
    # row = src.storage.row()
    # csr2csc = src.storage.csr2csc()
    colptr = src.storage.colptr()
    # opt_value = value.view(-1, 1).index_select(0, csr2csc).view(-1)
    row_T = src.storage.row_T()
    value_T = src.storage.value_T()

    row_splits, col_splits = src.get_work_range_for_transpose()
    if row_splits.size(0) == 0:
        num_threads = torch.get_num_threads()
        row_splits = distribute_work_for_row(colptr, value_T, num_threads)

    col_splits = disribute_work_for_col(other, 1)
    
    src.set_work_range_for_transpose(row_splits, col_splits)

    # return spmm_sum_without_backward(colptr, row.index_select(0, csr2csc), opt_value, other)
    return spmm_sum_without_backward(colptr, row_T, value_T, other, out, row_splits, col_splits)