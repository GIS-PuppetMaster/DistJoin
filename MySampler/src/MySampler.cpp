#include "MySampler.h"

using namespace torch::indexing;
// namespace py = pybind11;
// using Tensor = torch::Tensor;

// 线程池类
class ThreadPool {
public:
    // 获取全局单例线程池
    static ThreadPool& getInstance(size_t threads = std::thread::hardware_concurrency()) {
        static ThreadPool instance(threads);
        return instance;
    }

    // 禁止拷贝和赋值
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // 提交任务到线程池
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

private:
    // 构造函数私有化，确保单例
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

    std::vector<std::thread> workers;          // 工作线程
    std::queue<std::function<void()>> tasks;   // 任务队列
    std::mutex queue_mutex;                    // 任务队列互斥锁
    std::condition_variable condition;         // 条件变量
    bool stop;                                 // 线程池停止标志
};




std::vector<torch::Tensor>
sample_i(int i, const torch::Tensor &tuples, torch::Tensor *new_tuples_, torch::Tensor *new_preds_, int column_size,
         int start_idx, int end_idx, bool has_none, int num_samples, int first_pred, const torch::Tensor *first_mask, bool multi_preds, bool bounded_eqv) {
    int region_size = end_idx - start_idx;
    torch::TensorOptions device = torch::TensorOptions().device(tuples.device());
    // torch::Tensor pred_steps;
    torch::Tensor available_preds;
    torch::Tensor new_tuples;
    torch::Tensor new_preds;
    torch::Tensor pred_slices;
    int pred_idx;
    if (first_pred == -1) {
        torch::TensorOptions op = device.dtype(torch::kInt64).requires_grad(false);
        new_tuples = torch::zeros({num_samples, 2}, op) - 1;
        new_preds = torch::zeros({num_samples, 2, 5}, op);
        pred_idx = 0;
        int chunk_num = 5;
        if (bounded_eqv){
            available_preds = torch::tensor({0, }, op);
            chunk_num = 1;
        }
        else{
            available_preds = torch::randperm(5);
        }
        int step = region_size / chunk_num;
        pred_slices = torch::tensor(
                {start_idx, start_idx + step, start_idx + 2 * step, start_idx + 3 * step, start_idx + 4 * step,
                 end_idx}, device.dtype(torch::kInt64));

    } else {
        new_tuples = *new_tuples_;
        new_preds = *new_preds_;
        pred_idx = 1;
        int chunk_num = 3;
        if (first_pred == 1 || first_pred == 3) {
            // permutation [2,-1,4], start from permuting [0,1,2], then add 2, then replace 3 with -1
            available_preds = torch::randperm(3) + 2;
            torch::Tensor inava_indices = available_preds == 3;
            available_preds.index_put_({inava_indices}, -1);
        } else {
            // permutation [1, -1, 3], start from permuting [0,1,2], then add 1, then replace 2 with -1
            available_preds = torch::randperm(3) + 1;
            torch::Tensor inava_indices = available_preds == 2;
            available_preds.index_put_({inava_indices}, -1);
        }
        int step = region_size / chunk_num;
        pred_slices = torch::tensor({start_idx, start_idx + step, start_idx + 2 * step, end_idx},
                                    device.dtype(torch::kInt64));
       
    }
    torch::Tensor put_masks = torch::zeros({region_size,}, device.dtype(torch::kBool));
    torch::Tensor lower_bounds = torch::zeros({region_size,}, device);
    torch::Tensor higher_bounds = torch::zeros({region_size,}, device);
    for (int j = 0; j < available_preds.sizes()[0]; ++j) {
        int pred = available_preds.index({j}).item<int>();
        if (pred == -1) {
            continue;
        }
        int s = pred_slices.index({j}).item<int>();
        int e = pred_slices.index({j + 1}).item<int>();
        int sub_region_size = e - s;
        Slice ls = Slice(s, e);
        torch::Tensor local_tuples = tuples.index({ls});
        torch::Tensor local_new_tuples = new_tuples.index({ls, pred_idx});
        torch::Tensor local_new_preds = new_preds.index({ls, pred_idx, pred});
        torch::Tensor local_new_equal_preds = new_preds.index({ls, pred_idx, 0});
        if (pred == 0) {
            new_tuples.index_put_({ls, pred_idx}, local_tuples);
            local_new_preds.index_put_({"..."}, 1);
        } else {
            torch::Tensor lower_bound;
            torch::Tensor higher_bound;
            switch (pred) {
                case 1:
                case 3:
                    // cv >/>= pred
                    // assume a column has at least one non-NULL value
                    if (has_none) {
                        lower_bound = torch::ones({sub_region_size,}, device);
                    } else {
                        lower_bound = torch::zeros({sub_region_size,}, device);
                    }
                    if (pred == 1) {
                        higher_bound = local_tuples;
                        //std::cout<<"line 94, column_size:"<<column_size<<" high_bound.max():"<<higher_bound.max()<<std::endl;
                    } else {
                        higher_bound = local_tuples + 1;
                        //std::cout<<"line 97, column_size:"<<column_size<<" high_bound.max():"<<higher_bound.max()<<std::endl;
                    }
                    break;
                case 2:
                case 4:
                    if (pred == 2) {
                        lower_bound = local_tuples + 1;
                    } else {
                        lower_bound = local_tuples;
                        if (has_none) {
                            lower_bound.index_put_({lower_bound == 0}, 1);
                        }
                    }
                    // assume a column has at least one non-NULL value
                    higher_bound = torch::zeros({sub_region_size,}, device) + column_size;
                    //std::cout<<"line 112, column_size:"<<column_size<<" high_bound.max():"<<higher_bound.max()<<std::endl;
                    break;
            }
            torch::Tensor put_mask = lower_bound < higher_bound;
            Slice gs = Slice(s - start_idx, e - start_idx);
            if (first_mask != nullptr) {
                torch::bitwise_and_out(put_mask, put_mask,
                                       first_mask->index({Slice(s - start_idx, e - start_idx)}));
            }
            lower_bounds.index_put_({gs}, lower_bound);
            higher_bounds.index_put_({gs}, higher_bound);
            put_masks.index_put_({gs}, put_mask);
            local_new_preds.index_put_({put_mask}, 1);
            torch::Tensor un_mask = torch::logical_not(put_mask);
            if (first_pred == -1) {
                local_new_equal_preds.index_put_({un_mask}, 1);
                // local_new_tuples.index_put_({un_mask}, local_tuples.index({un_mask}));
                if (multi_preds && put_mask.any().item().toBool()) {
                    sample_i(i, tuples, &new_tuples, &new_preds, column_size, s, e, has_none, num_samples, pred,
                             &put_mask);
                }
            }
        }
    }
    Slice slice = Slice(start_idx, end_idx);
    torch::Tensor samples =
            (higher_bounds - lower_bounds) * torch::rand({region_size,}, device.dtype(torch::kFloat64)) + lower_bounds;
    torch::Tensor local_tuples = tuples.index({slice});
    torch::Tensor not_put_masks = torch::logical_not(put_masks);
    torch::Tensor local_new_tuples = new_tuples.index({slice, pred_idx});
    local_new_tuples.index_put_({put_masks}, torch::floor(samples.index({put_masks})).to(
            torch::TensorOptions().dtype(torch::kInt64)));
    if (first_pred == -1) {
        local_new_tuples.index_put_({not_put_masks}, local_tuples.index({not_put_masks}));
        // std::vector<torch::Tensor>&& tmp_ret = std::vector<torch::Tensor>{torch::tensor({i}), torch::unsqueeze(new_tuples, -1), new_preds};
        // return tmp_ret;
        return std::vector<torch::Tensor>{torch::tensor({i}), torch::unsqueeze(new_tuples, -1), new_preds};
    }
    return std::vector<torch::Tensor>{};
}

// 样本处理函数
void sample(const torch::Tensor &tuples, torch::Tensor &new_tuples, torch::Tensor &new_preds,
            const std::vector<int> &columns_size, std::vector<bool> &has_nones, int num_samples,
            bool multi_preds, int bounded_eqv_col_idx) {
    unsigned long long col_num = columns_size.size();
    std::vector<torch::Tensor> new_tuples_list(col_num);
    std::vector<torch::Tensor> new_preds_list(col_num);

    // 获取全局单例线程池
    ThreadPool& pool = ThreadPool::getInstance();

    // 提交任务到线程池
    std::vector<std::future<void>> futures;
    for (int i = 0; i < col_num; ++i) {
        futures.emplace_back(pool.enqueue([i, &tuples, &new_tuples_list, &new_preds_list, &columns_size, &has_nones, num_samples, multi_preds, bounded_eqv_col_idx] {
            auto ret = sample_i(i, tuples.index({"...", i}), nullptr, nullptr, columns_size[i], 0, num_samples,
                              has_nones[i], num_samples, -1, nullptr, multi_preds, i == bounded_eqv_col_idx);
            new_tuples_list[i] = ret[1];
            new_preds_list[i] = ret[2];
        }));
    }

    // 等待所有任务完成
    for (auto& future : futures) {
        future.wait();
    }

    // 合并结果
    torch::cat_out(new_tuples, at::TensorList(new_tuples_list), -1);
    torch::cat_out(new_preds, at::TensorList(new_preds_list), -1);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample", &sample, "do sample", py::arg("tuples"), py::arg("new_tuples"), py::arg("new_preds"),
          py::arg("columns_size"), py::arg("has_nones"), py::arg("num_samples"), py::arg("multi_preds"), py::arg("bounded_eqv_col_idx"));
}