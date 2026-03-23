# 基准测试与内部实现

---

## 大规模 CRUD 基准测试

测试环境：**9,000,000 条记录** · `dim=128` · L2 距离 · Windows AMD64 · Python 3.13 · 69 GB RAM。

完整基准脚本：[`benchmarks/benchmark_10m.py`](../benchmarks/benchmark_10m.py)

---

### IVF 索引——9M 完整 CRUD 基准

配置：`nlist=500`，`nprobe=50`（每次查询探测 10% 的聚类），10K 训练向量。

| 操作 | 记录数 | 耗时 | 吞吐量 |
|------|--------|------|--------|
| **INSERT**（训练后） | 9,000,000 | 2m 26s | **61,494 vec/s** |
| **SEARCH** k=10 | 200 次查询 | — | p50=**222 ms** · p95=245 ms · p99=258 ms · QPS=4 |
| **UPDATE** | 100,000 | 4m 35s | **363 ops/s** |
| **DELETE** | 100,000 | 6m 10s | **270 ops/s** |

**INSERT 里程碑：**

| 累计记录数 | 耗时 | 吞吐量 |
|-----------|------|--------|
| 1,000,000 | 16.67 s | 59,989 vec/s |
| 2,000,000 | 33.14 s | 60,355 vec/s |
| 3,000,000 | 49.85 s | 60,176 vec/s |
| 4,000,000 | 1m 06s | 60,120 vec/s |
| 5,000,000 | 1m 23s | 59,737 vec/s |
| 6,000,000 | 1m 40s | 59,938 vec/s |
| 7,000,000 | 1m 56s | 60,016 vec/s |
| 8,000,000 | 2m 12s | 60,387 vec/s |
| 9,000,000 | 2m 26s | 61,494 vec/s |

> IVF 插入是 O(1)——每条向量只需分配到最近的聚类质心，吞吐量在所有规模下几乎持平。`nprobe=50` 时的查询延迟对应于每次扫描 50 个聚类 × ~18K 条向量（9M 规模下约 90 万次距离计算）。UPDATE 和 DELETE 需要 O(N) 全表 id 扫描；写入密集型场景更适合使用 HNSW（惰性删除，无全表扫描）。

---

### HNSW 索引——大规模插入吞吐量

配置：`M=8`，`ef_construction=50`，`ef_search=32` · dim=128 · L2。

| 累计记录数 | 耗时 | 吞吐量 |
|-----------|------|--------|
| 1,000,000 | 2m 54s | 5,741 vec/s |
| 2,000,000 | 7m 20s | 4,540 vec/s |
| 3,000,000 | 11m 27s | 4,362 vec/s |
| 4,000,000 | 15m 49s | 4,212 vec/s |
| 5,000,000 | 21m 54s | 3,805 vec/s |
| 6,000,000 | 32m 02s | 3,122 vec/s |
| 7,000,000 | 44m 00s | 2,651 vec/s |
| 8,000,000 | 56m 03s | 2,378 vec/s |
| 9,000,000 | 1h 08m | 2,196 vec/s |

HNSW 插入代价随 N 对数增长（每次插入需以 O(M·log N) 次距离计算遍历图），因此吞吐量逐渐下降。作为回报，HNSW 在任意规模下均可实现**亚毫秒级近似 k-NN 查询**。

### 1000 万向量验证

HNSW 在 Windows AMD64 上成功插入 **10,000,000 条向量**（`dim=128`，`M=8`，`ef_construction=50`），全程无 OOM，得益于 VecStore 分块存储层的支持。

| 累计向量数 | 耗时 | 吞吐量 |
|-----------|------|--------|
| 500,000 | 94s | 5,314 vec/s |
| 1,000,000 | 3m 47s | 4,404 vec/s |
| 5,000,000 | 24m 53s | 3,348 vec/s |
| **10,000,000** | **1h 22m** | **2,030 vec/s** |

---

## SIMD 加速距离计算核

全部 5 种距离函数均配备了手写 SIMD 内核，**运行时自动选择**，无需任何配置：

| 架构 | ISA | 主循环策略 | 典型加速比 |
|---|---|---|---|
| x86-64（Haswell+） | **AVX2 + FMA** | 每次迭代 16 floats，双累加器展开，融合乘加 | 标量的 4–8× |
| ARM / Apple Silicon | **NEON** | 每次迭代 16 floats（4× `float32x4_t`），4 累加器展开 | 标量的 3–5× |
| 其他架构 | 标量 | 标准 C11 循环 | 基准线 |

**运行时探测**——首次调用时对 CPU 进行一次探测（GCC/Clang 使用 `__builtin_cpu_supports`，MSVC 使用 `__cpuid` + `_xgetbv`），随即将函数指针原地替换为最优实现，此后所有调用直接跳转，无分支开销。

| 计算核 | AVX2 技术 | NEON 技术 |
|---|---|---|
| `vec_dot` / `dist_ip` | `_mm256_fmadd_ps` × 2 YMM | `vmlaq_f32` × 4 Q-寄存器 |
| `dist_l2sq` / `dist_l2` | `_mm256_sub_ps` → `_mm256_fmadd_ps` | `vsubq_f32` → `vmlaq_f32` |
| `dist_cosine` | 单遍三路累加器（dot、‖a‖²、‖b‖²） | 6 个 Q-寄存器累加器 |
| `dist_l1` | 符号位掩码（`AND 0x7FFFFFFF`） | `vabsq_f32` → `vaddq_f32` |
| `dist_hamming` | `_mm256_cmp_ps` + `movemask` + `popcount` | `vceqq_f32` + `vshrq_n_u32` + `vaddvq_u32` |

---

## 文件格式

```
PistaDB 文件格式 v1.0
───────────────────────────────────────────
[文件头]  128 字节（固定）
  magic[4]           = "PSDB"
  version_major[2]   = 1
  version_minor[2]   = 0
  flags[4]           = 0（保留，供未来使用）
  dimension[4]       = 向量维度
  metric_type[2]     = 距离度量枚举值
  index_type[2]      = 索引算法枚举值
  num_vectors[8]     = 向量总数（含软删除）
  next_id[8]         = 自增 id 提示值
  vec_offset[8]      = 数据块起始字节偏移
  vec_size[8]        = 数据块大小（字节）
  idx_offset[8]      = 辅助索引数据起始偏移
  idx_size[8]        = 辅助索引数据大小
  reserved[54]       = 0
  header_crc[4]      = 文件头前 124 字节的 CRC32

[向量 + 索引数据]  位于 vec_offset
  各索引类型的自描述二进制序列化（版本化）
```

每种索引类型（HNSW、IVF、DiskANN、LSH、ScaNN……）都将自身序列化到该数据块中——图边、聚类质心、PQ 码本、投影矩阵及原始浮点向量全部存于一处。文件完全自包含：无附属文件，无注册表条目，加载时无需任何环境变量。

---

## 项目结构

```
PistaDB/
├── src/                          # C99 核心——所有平台共用
│   ├── pistadb_types.h           # 共享类型、错误码、默认参数
│   ├── pistadb.h / .c            # 主数据库 API
│   ├── distance.h / .c           # 运行时派发 + 5 种度量的标量回退
│   ├── distance_avx2.c           # AVX2+FMA 内核（-mavx2 -mfma 编译）
│   ├── distance_neon.c           # ARM NEON 内核（AArch64 内置）
│   ├── pistadb_batch.h / .c      # 多线程批量插入（线程池 + 环形队列）
│   ├── pistadb_cache.h / .c      # 嵌入缓存（FNV-1a 哈希表 + LRU + .pcc 文件）
│   ├── pistadb_txn.h / .c        # 事务 API（原子多操作，失败回滚）
│   ├── index_linear.*            # 精确暴力扫描
│   ├── index_hnsw.*              # HNSW（Malkov & Yashunin, 2018）
│   ├── index_ivf.*               # IVF k-means 聚类
│   ├── index_ivf_pq.*            # IVF + 乘积量化
│   ├── index_diskann.*           # Vamana / DiskANN（Subramanya et al., 2019）
│   ├── index_lsh.*               # E2LSH + 符号 LSH
│   └── index_scann.*             # ScaNN 各向异性向量量化（Guo et al., ICML 2020）
├── python/                       # Python 绑定（纯 ctypes，无 Cython）
├── go/                           # Go 绑定（CGO），模块：pistadb.io/go
├── android/                      # Android JNI 桥接 + Java + Kotlin API
├── ios/                          # iOS/macOS ObjC + Swift SPM 包
├── wasm/                         # WebAssembly 绑定（Emscripten + Embind）
├── cpp/                          # C++ 纯头文件封装（C++17）
├── rust/                         # Rust FFI crate（无外部依赖）
├── csharp/                       # C# P/Invoke 绑定（.NET Standard 2.0）
├── tests/
│   └── test_pistadb.py           # 109 项 pytest 测试套件
├── examples/
│   └── example.py                # 12 个端到端使用场景
├── benchmarks/
│   └── benchmark_10m.py          # 大规模 CRUD 基准测试脚本
├── Package.swift                 # SPM 清单文件
├── CMakeLists.txt
├── build.sh                      # Linux / macOS 构建脚本
└── build.bat                     # Windows MSVC 构建脚本
```
