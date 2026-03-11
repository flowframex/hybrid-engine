# Hybrid CPU-GPU Compute Engine

> Offloads math workloads from your CPU to CUDA — built for low-end hardware.  
> Target machine: Intel i3-2120 + NVIDIA GT730 (CUDA 8.0, sm_21/sm_35)

---

## What This Engine Actually Does

Your i3-2120 hits 100% CPU because it's doing **all the math alone**.  
This engine splits compute work:

```
Your workload
    │
    ▼
 Dispatcher
    ├── Is this a big parallel array / matrix op?
    │       └── YES → CUDA GPU kernels (GT730's 384 CUDA cores)
    │
    └── Is this small / branchy / serial logic?
            └── YES → CPU thread pool (4 logical cores)
```

The **Load Balancer** watches both sides every 200ms and shifts the ratio  
so neither the CPU nor GPU sits idle.

---

## Supported Operations

| Operation | CPU | GPU (CUDA kernel) |
|-----------|-----|-------------------|
| Vector add / sub / mul | ✓ | ✓ `k_vec_add` |
| Vector scale / sqrt / exp / log | ✓ | ✓ `k_vec_*` |
| sin / cos | ✓ | ✓ |
| Dot product | ✓ | ✓ 2-pass reduction |
| Reduce sum | ✓ | ✓ shared-mem reduction |
| Matrix multiply (tiled) | ✓ | ✓ `k_matmul_tiled` |
| Custom CPU lambda | ✓ | — |

---

## Building Locally (Windows, with CUDA installed)

```bash
# Install CUDA Toolkit 11.8 from https://developer.nvidia.com/cuda-11-8-0-download-archive
# Then open a CUDA developer command prompt:

nvcc hybrid_engine.cu -o hybrid_engine.exe ^
  -arch=sm_21 ^
  -gencode arch=compute_21,code=sm_21 ^
  -gencode arch=compute_35,code=sm_35 ^
  -O3 -std=c++17 -Xcompiler /O2 -w
```

**Which `-arch` do I use for my GT730?**
- If your GT730 says **GF108** on GPU-Z → use `sm_21` (Fermi)
- If your GT730 says **GK208** on GPU-Z → use `sm_35` (Kepler)
- The workflow compiles both, so the binary handles either.

---

## Building via GitHub Actions (No local CUDA needed)

1. Create a GitHub repo.
2. Put `hybrid_engine.cu` in the root.
3. Put `.github/workflows/build.yml` in the `.github/workflows/` folder.
4. Push. Actions auto-runs and produces:
   - `hybrid_engine_linux_x64` — Linux binary
   - `hybrid_engine_windows_x64` — Windows `.exe` for your PC
5. Download the artifact from the **Actions** tab → run it.

---

## Integrating Into Your App

```cpp
#include "hybrid_engine.cu"   // or compile separately and link

int main() {
    HybridEngine engine;      // auto-detects GPU

    // Big array? Goes to CUDA automatically:
    auto a = fill_with_game_data(1'000'000);
    auto b = fill_with_game_data(1'000'000);
    auto c = engine.vec_add(a, b);

    // Or use the macros — even simpler:
    auto result = ENGINE_ADD(a, b);
    float total = ENGINE_SUM(result);

    engine.print_stats();     // see GPU vs CPU split
}
```

---

## Why This Won't Directly Fix BlueStacks FPS

**Honest note:** BlueStacks's CPU bottleneck is ARM-to-x86 **instruction translation**,  
not floating-point math arrays. That translation is inherently serial — no GPU can  
accelerate it because each emulated instruction depends on the previous one's output.

**What this engine DOES help with:**
- Any C++ game/simulation/renderer you write yourself
- Physics engines, AI pathfinding, particle systems, signal processing
- Any app that does bulk math on arrays

**To actually help BlueStacks:** try **LDPlayer** or **MuMu Player** — they have  
better CPU emulation efficiency for the i3-2120 specifically.  
Also set BlueStacks to use 2 CPU cores + High Performance mode in its settings.

---

## Architecture

```
HybridEngine
├── CPUThreadPool      (4 threads, i3-2120 logical cores)
├── CUDAStreamPool     (4 async streams, GT730)
├── GPUExecutor        (launches kernels, syncs streams)
├── CPUExecutor        (std::thread work queue)
├── LoadBalancer       (200ms rebalance loop, adjusts gpu_ratio)
└── PerfStats          (atomic counters, avg latency tracking)
```
