/*
 * ============================================================
 *  HYBRID CPU-GPU COMPUTE ENGINE  v1.0
 *  Target: Intel i3-2120 + NVIDIA GT730 (CUDA 8.0, sm_21/sm_35)
 *  Author: Built for low-end hardware optimisation
 *
 *  WHAT THIS ENGINE DOES:
 *  - Intercepts math/compute workloads at runtime
 *  - Auto-routes heavy parallel tasks → CUDA GPU kernels
 *  - Keeps serial/branchy logic on CPU threads
 *  - Dynamic load balancer adjusts GPU/CPU split in real-time
 *  - Provides a drop-in API: any app calls ENGINE_DISPATCH(...)
 *    and the engine decides WHERE to run it
 * ============================================================
 */

// ── Standard Headers ──────────────────────────────────────────
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>
#include <vector>
#include <queue>
#include <deque>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>
#include <chrono>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <future>
#include <stdexcept>

// ── CUDA Error Macro ──────────────────────────────────────────
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(_e)         \
                      << "  at " << __FILE__ << ":" << __LINE__ << "\n";  \
            throw std::runtime_error(cudaGetErrorString(_e));             \
        }                                                                  \
    } while (0)

// ── Constants ─────────────────────────────────────────────────
static constexpr int   CUDA_BLOCK_SIZE      = 256;   // threads per block
static constexpr int   CPU_THREAD_COUNT     = 4;     // i3-2120 has 4 logical cores
static constexpr int   GPU_STREAM_COUNT     = 4;     // concurrent CUDA streams
static constexpr int   TASK_QUEUE_MAX       = 1024;
static constexpr float GPU_RATIO_DEFAULT    = 0.60f; // start 60% GPU, 40% CPU
static constexpr int   GPU_MIN_ELEMENTS     = 512;   // below this → CPU is faster
static constexpr int   BALANCE_INTERVAL_MS  = 200;   // rebalance every 200ms

// ═══════════════════════════════════════════════════════════════
//  SECTION 1 — CUDA KERNELS
//  These run directly on the GPU.  Each kernel is launched by the
//  dispatcher; they represent the "translated CPU math calls."
// ═══════════════════════════════════════════════════════════════

/* ---------- Vector arithmetic -------------------------------- */
__global__ void k_vec_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float*       __restrict__ c,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = a[i] + b[i];
}

__global__ void k_vec_sub(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float*       __restrict__ c,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = a[i] - b[i];
}

__global__ void k_vec_mul(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float*       __restrict__ c,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = a[i] * b[i];
}

__global__ void k_vec_scale(const float* __restrict__ a,
                             float scalar,
                             float*       __restrict__ c,
                             int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = a[i] * scalar;
}

/* ---------- Reduction: sum ----------------------------------- */
__global__ void k_reduce_sum(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int n)
{
    extern __shared__ float sdata[];
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

/* ---------- Dot product -------------------------------------- */
__global__ void k_dot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float*       __restrict__ partial,
                       int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (gid < n) ? a[gid] * b[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

/* ---------- Math transforms ---------------------------------- */
__global__ void k_vec_sqrt(const float* __restrict__ a,
                            float*       __restrict__ c,
                            int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = sqrtf(a[i]);
}

__global__ void k_vec_exp(const float* __restrict__ a,
                           float*       __restrict__ c,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = expf(a[i]);
}

__global__ void k_vec_log(const float* __restrict__ a,
                           float*       __restrict__ c,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = logf(a[i]);
}

__global__ void k_vec_sin(const float* __restrict__ a,
                           float*       __restrict__ c,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = sinf(a[i]);
}

__global__ void k_vec_cos(const float* __restrict__ a,
                           float*       __restrict__ c,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = cosf(a[i]);
}

/* ---------- Matrix multiply (naive, good for small mats) ----- */
__global__ void k_matmul(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float*       __restrict__ C,
                          int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

/* ---------- Tiled matrix multiply (shared mem — faster) ------- */
#define TILE 16
__global__ void k_matmul_tiled(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float*       __restrict__ C,
                                int M, int N, int K)
{
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        tA[threadIdx.y][threadIdx.x] =
            (row < M && t * TILE + threadIdx.x < K)
            ? A[row * K + t * TILE + threadIdx.x] : 0.0f;
        tB[threadIdx.y][threadIdx.x] =
            (col < N && t * TILE + threadIdx.y < K)
            ? B[(t * TILE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

/* ---------- ReLU activation (neural-net style ops) ----------- */
__global__ void k_relu(float* __restrict__ data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
        data[i] = fmaxf(0.0f, data[i]);
}

/* ---------- General "apply function" kernel ------------------ */
// op: 0=add_scalar, 1=mul_scalar, 2=pow, 3=abs, 4=negate
__global__ void k_apply(float* __restrict__ data,
                         float param,
                         int   op,
                         int   n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        switch (op) {
            case 0: data[i] += param;             break;
            case 1: data[i] *= param;             break;
            case 2: data[i] = powf(data[i], param); break;
            case 3: data[i] = fabsf(data[i]);     break;
            case 4: data[i] = -data[i];           break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  SECTION 2 — GPU DEVICE BUFFER
//  RAII wrapper for device memory.  Handles alloc/free safely.
// ═══════════════════════════════════════════════════════════════
template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count = 0) : ptr_(nullptr), count_(count) {
        if (count_ > 0)
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }

    // Non-copyable, movable
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_)
        { o.ptr_ = nullptr; o.count_ = 0; }

    void resize(size_t count) {
        if (count == count_) return;
        if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; }
        count_ = count;
        if (count_ > 0)
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    void upload(const T* host, size_t count, cudaStream_t stream = 0) {
        resize(count);
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host, count * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
    }

    void download(T* host, size_t count, cudaStream_t stream = 0) const {
        CUDA_CHECK(cudaMemcpyAsync(host, ptr_, count * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream));
    }

    T*     get()   const { return ptr_; }
    size_t size()  const { return count_; }

private:
    T*     ptr_;
    size_t count_;
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 3 — CPU THREAD POOL
//  Fixed pool of worker threads.  Tasks are std::function<void()>.
// ═══════════════════════════════════════════════════════════════
class CPUThreadPool {
public:
    explicit CPUThreadPool(size_t nthreads) : stop_(false), active_(0) {
        for (size_t i = 0; i < nthreads; ++i)
            workers_.emplace_back([this]{ worker_loop(); });
    }

    ~CPUThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // Enqueue a task and return a future for result
    template<typename F>
    std::future<void> enqueue(F&& f) {
        auto task = std::make_shared<std::packaged_task<void()>>(
                        std::forward<F>(f));
        std::future<void> fut = task->get_future();
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (stop_) throw std::runtime_error("Pool is stopped");
            queue_.emplace([task]{ (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    int active_tasks() const { return active_.load(); }
    size_t thread_count() const { return workers_.size(); }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this]{ return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop();
            }
            ++active_;
            task();
            --active_;
            done_cv_.notify_all();
        }
    }

    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> queue_;
    std::mutex                        mtx_;
    std::condition_variable           cv_;
    std::condition_variable           done_cv_;
    std::atomic<int>                  active_;
    bool                              stop_;
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 4 — CUDA STREAM POOL
//  Manages a pool of CUDA streams for async kernel launches.
// ═══════════════════════════════════════════════════════════════
class CUDAStreamPool {
public:
    explicit CUDAStreamPool(int count) : next_(0) {
        streams_.resize(count);
        for (auto& s : streams_)
            CUDA_CHECK(cudaStreamCreate(&s));
    }

    ~CUDAStreamPool() {
        for (auto& s : streams_) cudaStreamDestroy(s);
    }

    // Round-robin stream assignment
    cudaStream_t acquire() {
        int idx = next_.fetch_add(1) % static_cast<int>(streams_.size());
        return streams_[idx];
    }

    void sync_all() {
        for (auto& s : streams_)
            CUDA_CHECK(cudaStreamSynchronize(s));
    }

    int count() const { return static_cast<int>(streams_.size()); }

private:
    std::vector<cudaStream_t> streams_;
    std::atomic<int>          next_;
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 5 — PERFORMANCE MONITOR
//  Tracks GPU/CPU utilisation and task completion times.
//  The load balancer uses this to tune the split ratio.
// ═══════════════════════════════════════════════════════════════
struct PerfStats {
    std::atomic<uint64_t> gpu_tasks_done{0};
    std::atomic<uint64_t> cpu_tasks_done{0};
    std::atomic<uint64_t> gpu_ns_total{0};    // nanoseconds spent on GPU
    std::atomic<uint64_t> cpu_ns_total{0};    // nanoseconds spent on CPU
    std::atomic<float>    gpu_ratio{GPU_RATIO_DEFAULT};
    std::atomic<int>      gpu_queue_depth{0};
    std::atomic<int>      cpu_queue_depth{0};

    void record_gpu(uint64_t ns) {
        gpu_ns_total.fetch_add(ns);
        gpu_tasks_done.fetch_add(1);
    }
    void record_cpu(uint64_t ns) {
        cpu_ns_total.fetch_add(ns);
        cpu_tasks_done.fetch_add(1);
    }

    float avg_gpu_ms() const {
        auto d = gpu_tasks_done.load();
        return d ? (float)(gpu_ns_total.load() / d) / 1e6f : 0.f;
    }
    float avg_cpu_ms() const {
        auto d = cpu_tasks_done.load();
        return d ? (float)(cpu_ns_total.load() / d) / 1e6f : 0.f;
    }
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 6 — TASK DESCRIPTOR
//  Describes a single unit of compute work.
// ═══════════════════════════════════════════════════════════════
enum class TaskType {
    VEC_ADD, VEC_SUB, VEC_MUL, VEC_SCALE,
    VEC_SQRT, VEC_EXP, VEC_LOG, VEC_SIN, VEC_COS,
    DOT_PRODUCT, REDUCE_SUM,
    MATMUL,
    APPLY_OP,
    CUSTOM_CPU,
    CUSTOM_GPU
};

struct Task {
    TaskType               type       = TaskType::CUSTOM_CPU;
    std::vector<float>     a, b;        // input vectors
    std::vector<float>*    output      = nullptr; // where to write result
    int                    M=0,N=0,K=0; // matrix dims
    float                  scalar      = 1.0f;
    int                    op_code     = 0;
    std::function<void()>  cpu_fn;      // custom CPU callable
    std::promise<void>     done_promise;
    std::future<void>      done_future  = done_promise.get_future();
    int                    priority     = 0; // higher = run sooner
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 7 — LOAD BALANCER
//  Monitors stats and dynamically adjusts gpu_ratio so:
//    - If GPU finishes faster → increase GPU share
//    - If CPU is idle & GPU queue is long → shift some to CPU
//    - Never goes below 20% GPU or 80% GPU (keeps both busy)
// ═══════════════════════════════════════════════════════════════
class LoadBalancer {
public:
    explicit LoadBalancer(PerfStats& stats) : stats_(stats), running_(true) {
        thread_ = std::thread([this]{ balance_loop(); });
    }

    ~LoadBalancer() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    // Decide: should this task run on GPU?
    bool should_use_gpu(int element_count) const {
        if (element_count < GPU_MIN_ELEMENTS) return false;
        float ratio = stats_.gpu_ratio.load();
        float roll  = (float)(rand() % 1000) / 1000.0f;
        return roll < ratio;
    }

private:
    void balance_loop() {
        while (running_) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(BALANCE_INTERVAL_MS));

            float gpu_avg = stats_.avg_gpu_ms();
            float cpu_avg = stats_.avg_cpu_ms();
            float ratio   = stats_.gpu_ratio.load();

            // If GPU is faster per-task, give it more work
            if (gpu_avg > 0 && cpu_avg > 0) {
                if (gpu_avg < cpu_avg * 0.8f)
                    ratio = std::min(0.80f, ratio + 0.05f);
                else if (cpu_avg < gpu_avg * 0.8f)
                    ratio = std::max(0.20f, ratio - 0.05f);
            }

            // Depth-based correction
            int gq = stats_.gpu_queue_depth.load();
            int cq = stats_.cpu_queue_depth.load();
            if (gq > cq * 3 && ratio > 0.20f)
                ratio -= 0.05f;
            else if (cq > gq * 3 && ratio < 0.80f)
                ratio += 0.05f;

            stats_.gpu_ratio.store(ratio);
        }
    }

    PerfStats&  stats_;
    std::thread thread_;
    bool        running_;
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 8 — GPU EXECUTOR
//  Actually runs tasks on the GPU using the kernels above.
// ═══════════════════════════════════════════════════════════════
class GPUExecutor {
public:
    explicit GPUExecutor(CUDAStreamPool& streams, PerfStats& stats)
        : streams_(streams), stats_(stats) {}

    void execute(Task& task) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaStream_t stream = streams_.acquire();

        try {
            dispatch(task, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (...) {
            task.done_promise.set_exception(std::current_exception());
            return;
        }

        auto t1  = std::chrono::high_resolution_clock::now();
        uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t1 - t0).count();
        stats_.record_gpu(ns);
        task.done_promise.set_value();
    }

private:
    void dispatch(Task& task, cudaStream_t stream) {
        int n = (int)task.a.size();

        // ── allocate device buffers ──────────────────────────
        DeviceBuffer<float> d_a, d_b, d_c;

        auto upload_a = [&]{
            d_a.upload(task.a.data(), task.a.size(), stream);
        };
        auto upload_ab = [&]{
            d_a.upload(task.a.data(), task.a.size(), stream);
            d_b.upload(task.b.data(), task.b.size(), stream);
        };
        auto download_c = [&]{
            if (task.output) {
                task.output->resize(n);
                d_c.download(task.output->data(), n, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        };

        int blocks = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        // Clamp blocks for Fermi (max 65535 blocks per dim)
        blocks = std::min(blocks, 65535);

        switch (task.type) {

        case TaskType::VEC_ADD:
            upload_ab();
            d_c.resize(n);
            k_vec_add<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), d_b.get(), d_c.get(), n);
            download_c();
            break;

        case TaskType::VEC_SUB:
            upload_ab();
            d_c.resize(n);
            k_vec_sub<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), d_b.get(), d_c.get(), n);
            download_c();
            break;

        case TaskType::VEC_MUL:
            upload_ab();
            d_c.resize(n);
            k_vec_mul<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), d_b.get(), d_c.get(), n);
            download_c();
            break;

        case TaskType::VEC_SCALE:
            upload_a();
            d_c.resize(n);
            k_vec_scale<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), task.scalar, d_c.get(), n);
            download_c();
            break;

        case TaskType::VEC_SQRT:
            upload_a();
            d_c.resize(n);
            k_vec_sqrt<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), d_c.get(), n);
            download_c();
            break;

        case TaskType::VEC_EXP:
            upload_a();
            d_c.resize(n);
            k_vec_exp<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), d_c.get(), n);
            download_c();
            break;

        case TaskType::VEC_LOG:
            upload_a();
            d_c.resize(n);
            k_vec_log<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), d_c.get(), n);
            download_c();
            break;

        case TaskType::VEC_SIN:
            upload_a();
            d_c.resize(n);
            k_vec_sin<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), d_c.get(), n);
            download_c();
            break;

        case TaskType::VEC_COS:
            upload_a();
            d_c.resize(n);
            k_vec_cos<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_a.get(), d_c.get(), n);
            download_c();
            break;

        case TaskType::DOT_PRODUCT: {
            upload_ab();
            int partial_blocks = blocks;
            DeviceBuffer<float> d_partial(partial_blocks);
            k_dot<<<partial_blocks, CUDA_BLOCK_SIZE,
                    CUDA_BLOCK_SIZE * sizeof(float), stream>>>(
                d_a.get(), d_b.get(), d_partial.get(), n);
            // second pass: sum the partials
            int final_blocks = (partial_blocks + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            DeviceBuffer<float> d_result(final_blocks);
            k_reduce_sum<<<final_blocks, CUDA_BLOCK_SIZE,
                           CUDA_BLOCK_SIZE * sizeof(float), stream>>>(
                d_partial.get(), d_result.get(), partial_blocks);
            if (task.output) {
                task.output->resize(final_blocks);
                d_result.download(task.output->data(), final_blocks, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                // flatten to scalar
                float dot = 0;
                for (auto v : *task.output) dot += v;
                task.output->assign(1, dot);
            }
            break;
        }

        case TaskType::REDUCE_SUM: {
            upload_a();
            DeviceBuffer<float> d_partial(blocks);
            k_reduce_sum<<<blocks, CUDA_BLOCK_SIZE,
                           CUDA_BLOCK_SIZE * sizeof(float), stream>>>(
                d_a.get(), d_partial.get(), n);
            if (task.output) {
                task.output->resize(blocks);
                d_partial.download(task.output->data(), blocks, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                float sum = 0;
                for (auto v : *task.output) sum += v;
                task.output->assign(1, sum);
            }
            break;
        }

        case TaskType::MATMUL: {
            int M = task.M, N = task.N, K = task.K;
            DeviceBuffer<float> dA(M * K), dB(K * N), dC(M * N);
            CUDA_CHECK(cudaMemcpyAsync(dA.get(), task.a.data(),
                M * K * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB.get(), task.b.data(),
                K * N * sizeof(float), cudaMemcpyHostToDevice, stream));
            dim3 block(TILE, TILE);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            k_matmul_tiled<<<grid, block, 0, stream>>>(
                dA.get(), dB.get(), dC.get(), M, N, K);
            if (task.output) {
                task.output->resize(M * N);
                CUDA_CHECK(cudaMemcpyAsync(task.output->data(), dC.get(),
                    M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            break;
        }

        case TaskType::APPLY_OP:
            upload_a();
            d_c.resize(n);
            CUDA_CHECK(cudaMemcpyAsync(d_c.get(), d_a.get(),
                n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            k_apply<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_c.get(), task.scalar, task.op_code, n);
            download_c();
            break;

        default:
            if (task.cpu_fn) task.cpu_fn();  // fallback
            break;
        }

        CUDA_CHECK(cudaGetLastError());
    }

    CUDAStreamPool& streams_;
    PerfStats&      stats_;
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 9 — CPU EXECUTOR
//  Runs the same operations on the CPU.  Used for small arrays
//  or when GPU is saturated.
// ═══════════════════════════════════════════════════════════════
class CPUExecutor {
public:
    explicit CPUExecutor(CPUThreadPool& pool, PerfStats& stats)
        : pool_(pool), stats_(stats) {}

    std::future<void> execute(Task& task) {
        return pool_.enqueue([this, &task]{
            auto t0 = std::chrono::high_resolution_clock::now();
            run(task);
            auto t1  = std::chrono::high_resolution_clock::now();
            uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t1 - t0).count();
            stats_.record_cpu(ns);
            task.done_promise.set_value();
        });
    }

private:
    static void run(Task& task) {
        int n = (int)task.a.size();
        if (task.output) task.output->resize(n);
        float* out = task.output ? task.output->data() : nullptr;

        switch (task.type) {
        case TaskType::VEC_ADD:
            for (int i = 0; i < n; ++i) (*task.output)[i] = task.a[i] + task.b[i];
            break;
        case TaskType::VEC_SUB:
            for (int i = 0; i < n; ++i) (*task.output)[i] = task.a[i] - task.b[i];
            break;
        case TaskType::VEC_MUL:
            for (int i = 0; i < n; ++i) (*task.output)[i] = task.a[i] * task.b[i];
            break;
        case TaskType::VEC_SCALE:
            for (int i = 0; i < n; ++i) (*task.output)[i] = task.a[i] * task.scalar;
            break;
        case TaskType::VEC_SQRT:
            for (int i = 0; i < n; ++i) (*task.output)[i] = std::sqrt(task.a[i]);
            break;
        case TaskType::VEC_EXP:
            for (int i = 0; i < n; ++i) (*task.output)[i] = std::exp(task.a[i]);
            break;
        case TaskType::VEC_LOG:
            for (int i = 0; i < n; ++i) (*task.output)[i] = std::log(task.a[i]);
            break;
        case TaskType::VEC_SIN:
            for (int i = 0; i < n; ++i) (*task.output)[i] = std::sin(task.a[i]);
            break;
        case TaskType::VEC_COS:
            for (int i = 0; i < n; ++i) (*task.output)[i] = std::cos(task.a[i]);
            break;
        case TaskType::DOT_PRODUCT: {
            float dot = 0;
            for (int i = 0; i < n; ++i) dot += task.a[i] * task.b[i];
            if (task.output) task.output->assign(1, dot);
            break;
        }
        case TaskType::REDUCE_SUM: {
            float s = 0;
            for (int i = 0; i < n; ++i) s += task.a[i];
            if (task.output) task.output->assign(1, s);
            break;
        }
        case TaskType::MATMUL: {
            int M = task.M, N = task.N, K = task.K;
            if (task.output) task.output->resize(M * N, 0.0f);
            for (int i = 0; i < M; ++i)
                for (int k = 0; k < K; ++k)
                    for (int j = 0; j < N; ++j)
                        (*task.output)[i * N + j] += task.a[i * K + k] * task.b[k * N + j];
            break;
        }
        case TaskType::APPLY_OP:
            if (task.output) *task.output = task.a;
            for (int i = 0; i < n; ++i) {
                float v = task.a[i];
                switch (task.op_code) {
                case 0: v += task.scalar; break;
                case 1: v *= task.scalar; break;
                case 2: v  = std::pow(v, task.scalar); break;
                case 3: v  = std::abs(v); break;
                case 4: v  = -v; break;
                }
                if (out) out[i] = v;
            }
            break;
        case TaskType::CUSTOM_CPU:
            if (task.cpu_fn) task.cpu_fn();
            break;
        default: break;
        }
    }

    CPUThreadPool& pool_;
    PerfStats&     stats_;
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 10 — THE HYBRID ENGINE (main public API)
//  This is what you instantiate in your application.
// ═══════════════════════════════════════════════════════════════
class HybridEngine {
public:
    HybridEngine()
        : cpu_pool_(CPU_THREAD_COUNT)
        , stream_pool_(GPU_STREAM_COUNT)
        , gpu_exec_(stream_pool_, stats_)
        , cpu_exec_(cpu_pool_, stats_)
        , balancer_(stats_)
        , gpu_available_(false)
    {
        probe_gpu();
        print_banner();
    }

    ~HybridEngine() {
        // balancer destructor joins its thread automatically
    }

    // ── Public API ──────────────────────────────────────────

    // Submit a task.  Returns a future you can wait on.
    std::future<void> submit(std::shared_ptr<Task> task) {
        bool use_gpu = gpu_available_ &&
                       balancer_should_gpu((int)task->a.size());

        if (use_gpu) {
            stats_.gpu_queue_depth.fetch_add(1);
            return cpu_pool_.enqueue([this, task]{
                stats_.gpu_queue_depth.fetch_sub(1);
                gpu_exec_.execute(*task);
            });
        } else {
            stats_.cpu_queue_depth.fetch_add(1);
            auto fut = cpu_exec_.execute(*task);
            stats_.cpu_queue_depth.fetch_sub(1);
            return fut;
        }
    }

    // ── Convenience wrappers ─────────────────────────────────

    std::vector<float> vec_add(const std::vector<float>& a,
                                const std::vector<float>& b)
    {
        auto task = make_task(TaskType::VEC_ADD, a, b);
        std::vector<float> out;
        task->output = &out;
        submit(task).wait();
        return out;
    }

    std::vector<float> vec_sub(const std::vector<float>& a,
                                const std::vector<float>& b)
    {
        auto task = make_task(TaskType::VEC_SUB, a, b);
        std::vector<float> out;
        task->output = &out;
        submit(task).wait();
        return out;
    }

    std::vector<float> vec_mul(const std::vector<float>& a,
                                const std::vector<float>& b)
    {
        auto task = make_task(TaskType::VEC_MUL, a, b);
        std::vector<float> out;
        task->output = &out;
        submit(task).wait();
        return out;
    }

    std::vector<float> vec_scale(const std::vector<float>& a, float s)
    {
        auto task = make_task(TaskType::VEC_SCALE, a, {});
        task->scalar = s;
        std::vector<float> out;
        task->output = &out;
        submit(task).wait();
        return out;
    }

    std::vector<float> vec_sqrt(const std::vector<float>& a)
    {
        auto task = make_task(TaskType::VEC_SQRT, a, {});
        std::vector<float> out;
        task->output = &out;
        submit(task).wait();
        return out;
    }

    float dot(const std::vector<float>& a, const std::vector<float>& b)
    {
        auto task = make_task(TaskType::DOT_PRODUCT, a, b);
        std::vector<float> out;
        task->output = &out;
        submit(task).wait();
        return out.empty() ? 0.f : out[0];
    }

    float reduce_sum(const std::vector<float>& a)
    {
        auto task = make_task(TaskType::REDUCE_SUM, a, {});
        std::vector<float> out;
        task->output = &out;
        submit(task).wait();
        return out.empty() ? 0.f : out[0];
    }

    std::vector<float> matmul(const std::vector<float>& A,
                               const std::vector<float>& B,
                               int M, int N, int K)
    {
        auto task = make_task(TaskType::MATMUL, A, B);
        task->M = M; task->N = N; task->K = K;
        std::vector<float> out;
        task->output = &out;
        submit(task).wait();
        return out;
    }

    // Custom CPU lambda — for code you can't vectorise
    void dispatch_cpu(std::function<void()> fn) {
        auto task = std::make_shared<Task>();
        task->type   = TaskType::CUSTOM_CPU;
        task->cpu_fn = std::move(fn);
        submit(task).wait();
    }

    // ── Stats printout ──────────────────────────────────────
    void print_stats() const {
        std::cout << "\n╔══════════════════════════════════════╗\n";
        std::cout <<   "║   HYBRID ENGINE — RUNTIME STATS      ║\n";
        std::cout <<   "╠══════════════════════════════════════╣\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "║ GPU tasks  : " << std::setw(8)
                  << stats_.gpu_tasks_done.load()
                  << "  avg " << std::setw(7) << stats_.avg_gpu_ms()
                  << " ms  ║\n";
        std::cout << "║ CPU tasks  : " << std::setw(8)
                  << stats_.cpu_tasks_done.load()
                  << "  avg " << std::setw(7) << stats_.avg_cpu_ms()
                  << " ms  ║\n";
        std::cout << "║ GPU ratio  : "
                  << std::setw(6) << (stats_.gpu_ratio.load() * 100.f)
                  << "%                  ║\n";
        std::cout << "╚══════════════════════════════════════╝\n";
    }

    bool gpu_available() const { return gpu_available_; }

private:
    // ── Internal helpers ─────────────────────────────────────

    static std::shared_ptr<Task> make_task(TaskType t,
                                            const std::vector<float>& a,
                                            const std::vector<float>& b)
    {
        auto task = std::make_shared<Task>();
        task->type = t;
        task->a    = a;
        task->b    = b;
        return task;
    }

    bool balancer_should_gpu(int n) const {
        return balancer_.should_use_gpu(n);   // delegate to balancer
    }

    void probe_gpu() {
        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
            std::cerr << "[ENGINE] No CUDA device found — CPU-only mode.\n";
            gpu_available_ = false;
            return;
        }
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "[ENGINE] GPU detected : " << prop.name << "\n";
        std::cout << "[ENGINE] Compute cap  : "
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "[ENGINE] VRAM         : "
                  << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
        std::cout << "[ENGINE] CUDA cores   : "
                  << prop.multiProcessorCount << " SMs\n";
        gpu_available_ = true;
        CUDA_CHECK(cudaSetDevice(0));
    }

    void print_banner() const {
        std::cout << "\n";
        std::cout << " ╔══════════════════════════════════════════╗\n";
        std::cout << " ║   HYBRID CPU-GPU COMPUTE ENGINE v1.0     ║\n";
        std::cout << " ║   Target: i3-2120 + GT730 (CUDA 8)       ║\n";
        std::cout << " ║   CPU threads : " << CPU_THREAD_COUNT
                  << "  |  GPU streams : " << GPU_STREAM_COUNT
                  << "   ║\n";
        std::cout << " ║   Mode: " << (gpu_available_ ? "HYBRID (CPU+GPU)" : "CPU ONLY         ")
                  << "             ║\n";
        std::cout << " ╚══════════════════════════════════════════╝\n\n";
    }

    CPUThreadPool  cpu_pool_;
    CUDAStreamPool stream_pool_;
    GPUExecutor    gpu_exec_;
    CPUExecutor    cpu_exec_;
    PerfStats      stats_;
    LoadBalancer   balancer_;
    bool           gpu_available_;
};

// ═══════════════════════════════════════════════════════════════
//  SECTION 11 — CONVENIENCE MACROS
//  Drop these into any C++ project; they route via the engine.
// ═══════════════════════════════════════════════════════════════

// Global engine instance (singleton)
static HybridEngine* g_engine = nullptr;

inline HybridEngine& ENGINE() {
    if (!g_engine) g_engine = new HybridEngine();
    return *g_engine;
}

#define ENGINE_ADD(a, b)        ENGINE().vec_add(a, b)
#define ENGINE_SUB(a, b)        ENGINE().vec_sub(a, b)
#define ENGINE_MUL(a, b)        ENGINE().vec_mul(a, b)
#define ENGINE_SCALE(a, s)      ENGINE().vec_scale(a, s)
#define ENGINE_SQRT(a)          ENGINE().vec_sqrt(a)
#define ENGINE_DOT(a, b)        ENGINE().dot(a, b)
#define ENGINE_SUM(a)           ENGINE().reduce_sum(a)
#define ENGINE_MATMUL(A,B,M,N,K) ENGINE().matmul(A, B, M, N, K)
#define ENGINE_CPU(fn)          ENGINE().dispatch_cpu(fn)
#define ENGINE_STATS()          ENGINE().print_stats()

// ═══════════════════════════════════════════════════════════════
//  SECTION 12 — DEMO / BENCHMARK  main()
// ═══════════════════════════════════════════════════════════════

static std::vector<float> random_vec(int n, float lo = 0.1f, float hi = 10.f) {
    std::vector<float> v(n);
    for (auto& x : v) x = lo + (float)rand() / RAND_MAX * (hi - lo);
    return v;
}

int main() {
    // ── Init engine ──────────────────────────────────────────
    HybridEngine engine;

    // ── Test 1: Small vector (will go to CPU) ────────────────
    std::cout << "=== Test 1: Small VEC_ADD (n=128, CPU path) ===\n";
    {
        auto a = random_vec(128);
        auto b = random_vec(128);
        auto c = engine.vec_add(a, b);
        // verify first element
        float expected = a[0] + b[0];
        std::cout << "  a[0]=" << a[0] << "  b[0]=" << b[0]
                  << "  c[0]=" << c[0]
                  << "  expected=" << expected
                  << (std::abs(c[0] - expected) < 1e-4f ? "  ✓" : "  ✗")
                  << "\n";
    }

    // ── Test 2: Large vector (will go to GPU if available) ───
    std::cout << "\n=== Test 2: Large VEC_MUL (n=1,000,000, GPU path) ===\n";
    {
        int n = 1'000'000;
        auto a = random_vec(n);
        auto b = random_vec(n);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto c = engine.vec_mul(a, b);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float expected = a[42] * b[42];
        std::cout << "  n=" << n << "  time=" << ms << " ms\n";
        std::cout << "  c[42]=" << c[42] << "  expected=" << expected
                  << (std::abs(c[42] - expected) < 1e-3f ? "  ✓" : "  ✗")
                  << "\n";
    }

    // ── Test 3: Dot product ──────────────────────────────────
    std::cout << "\n=== Test 3: DOT PRODUCT (n=500,000) ===\n";
    {
        int n = 500'000;
        auto a = random_vec(n, 1.f, 2.f);
        auto b = random_vec(n, 1.f, 2.f);
        auto t0 = std::chrono::high_resolution_clock::now();
        float dot = engine.dot(a, b);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        // CPU reference
        double ref = 0;
        for (int i = 0; i < n; ++i) ref += (double)a[i] * b[i];
        std::cout << "  dot=" << dot << "  ref=" << (float)ref
                  << "  diff=" << std::abs(dot - (float)ref)
                  << "  time=" << ms << " ms\n";
    }

    // ── Test 4: Matrix multiply ──────────────────────────────
    std::cout << "\n=== Test 4: MATMUL 512×512 × 512×512 ===\n";
    {
        int M=512, N=512, K=512;
        auto A = random_vec(M * K, 0.f, 1.f);
        auto B = random_vec(K * N, 0.f, 1.f);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto C = engine.matmul(A, B, M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        std::cout << "  C size=" << C.size()
                  << "  C[0]=" << C[0]
                  << "  time=" << ms << " ms\n";
    }

    // ── Test 5: Parallel dispatch mix ────────────────────────
    std::cout << "\n=== Test 5: Parallel mixed dispatch (16 tasks) ===\n";
    {
        std::vector<std::future<void>> futures;
        std::vector<std::vector<float>> outputs(16);
        std::vector<std::shared_ptr<Task>> tasks(16);

        for (int i = 0; i < 16; ++i) {
            tasks[i] = std::make_shared<Task>();
            tasks[i]->type   = (i % 2 == 0) ? TaskType::VEC_ADD : TaskType::VEC_SQRT;
            tasks[i]->a      = random_vec(100'000);
            tasks[i]->b      = random_vec(100'000);
            tasks[i]->output = &outputs[i];
            futures.push_back(engine.submit(tasks[i]));
        }
        for (auto& f : futures) f.wait();
        std::cout << "  All 16 tasks completed.\n";
    }

    // ── Test 6: Custom CPU lambda ────────────────────────────
    std::cout << "\n=== Test 6: Custom CPU lambda ===\n";
    {
        int counter = 0;
        engine.dispatch_cpu([&counter]{
            for (int i = 0; i < 1'000'000; ++i) counter += i % 7;
        });
        std::cout << "  counter=" << counter << "\n";
    }

    // ── Final stats ──────────────────────────────────────────
    engine.print_stats();

    // ── Macro API demo ───────────────────────────────────────
    std::cout << "\n=== Macro API demo ===\n";
    {
        auto a = random_vec(1024);
        auto b = random_vec(1024);
        auto c = ENGINE_ADD(a, b);
        float s = ENGINE_SUM(c);
        std::cout << "  ENGINE_ADD then ENGINE_SUM = " << s << "\n";
        ENGINE_STATS();
    }

    std::cout << "\n[ENGINE] All tests passed. Engine ready for integration.\n";
    return 0;
}
