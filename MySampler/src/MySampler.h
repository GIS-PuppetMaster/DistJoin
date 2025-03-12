#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

std::vector<torch::Tensor>
sample_i(int i, const torch::Tensor &tuples, torch::Tensor *new_tuples_, torch::Tensor *new_preds_, int columns_size,
         int start_idx, int end_idx, bool has_none, int num_samples, int first_pred = -1,
         const torch::Tensor *first_mask = nullptr, bool multi_preds = true, bool bounded_eqv=false);

void
sample(const torch::Tensor &tuples, torch::Tensor &new_tuples, torch::Tensor &new_preds,
       const std::vector<int> &columns_size, std::vector<bool> &has_nones, int num_samples, bool multi_preds, int bounded_eqv_col_idx);
