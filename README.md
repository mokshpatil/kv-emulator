# KV-SSD Emulator

A lightweight, user-space KV-SSD emulator for evaluating key-value mapping table strategies on flash storage devices. Built to replicate and extend the evaluation environment described in the KVPack paper (IEEE Transactions on Computers, 2025).

## Project Goal

Reproduce the KVPack evaluation environment and use it as a foundation for researching **learning-guided dynamic inlining policies** for KV-SSD mapping tables. Specifically:

1. Replicate KVPack-S (static inlining) and KVPack-D (dynamic inlining) results
2. Replace heuristic-based inlining decisions with lightweight ML models (linear regression, contextual bandits)
3. Evaluate under realistic KV workloads from production traces
4. Compare ML-based policies against KVPack-S, KVPack-D, and baseline hash table

## Background

### The Problem

KV-SSDs use a Key-to-Physical (K2P) mapping table to translate variable-length keys to physical flash page addresses. This table is too large to fit in the SSD's limited DRAM, so it spills to flash as **translation pages**. A **Cached Mapping Table (CMT)** caches recently accessed entries in DRAM, but CMT misses are expensive: each miss causes at least one extra flash read to fetch the translation page, on top of the data page read. For random workloads, CMT miss rates can exceed 80-91%.

### KVPack's Solution

KVPack (Saha et al., IEEE TC 2025) proposes inlining small KV pairs directly into mapping entries within translation pages. When a CMT miss occurs for an inlined entry, the value is already in the translation page -- eliminating the second flash read entirely.

- **KVPack-S**: Static inline threshold, set once via profiling (up to 61.5% read latency reduction)
- **KVPack-D**: Dynamic threshold that adapts to workload changes via continuous profiling (up to 42.8% read latency reduction)

### Our Extension

KVPack uses heuristic profiling (popular frame count tracking) to decide inline thresholds. We aim to replace this with ML-based decision logic that can learn from workload patterns and make finer-grained inlining decisions.

## Why Build a Custom Emulator

KVPack was evaluated on MoKE, a QEMU-based KV-SSD emulator built on top of FEMU. We evaluated three alternative paths before deciding to build from scratch:

| Platform | Verdict | Reason |
|----------|---------|--------|
| OpenMPDK KVSSD | Not viable | API-level emulation only. No FTL, no mapping tables, no flash modeling. Zero overlap with our requirements. |
| FEMU | Overkill | Provides flash timing model, but all KV-SSD logic (K2P, CMT, translation pages, inline entries) must be written from scratch. QEMU adds debugging complexity without commensurate benefit for mapping-level research. |
| MoKE | Ideal but unavailable | Not publicly released as a standalone project. Source code is not accessible. |

**Decision: build a lightweight Python emulator from scratch.** Rationale:

- All KV-SSD-specific logic must be written from scratch regardless of base platform
- Our key metrics (flash read counts, CMT hit/miss, inline ratio, WAF) do not require cycle-accurate flash timing
- A standalone user-space emulator is dramatically easier to develop, debug, and iterate on
- Python enables straightforward ML model integration (scikit-learn, numpy)
- Estimated 2,000-5,000 lines of well-structured Python

## Architecture

```
+----------------------------------------------+
|  Trace Replay Engine                          |
|  - Twitter cache traces (cluster 15, etc.)    |
|  - Facebook RocksDB traces (UDB, ZippyDB)    |
|  - Synthetic workload generator               |
+----------------------------------------------+
           | put(key, value) / get(key) / delete(key)
           v
+----------------------------------------------+
|  KV-SSD Firmware Emulator                     |
|  +----------------------------------------+  |
|  |  KV Command Dispatcher                 |  |
|  |  hash(key) -> route to mapping layer   |  |
|  +----------------------------------------+  |
|  |  GMD (Global Mapping Directory)        |  |
|  |  in-DRAM hash: key_hash -> tp_id       |  |
|  |  size: ~2.5MB for 64GB SSD config      |  |
|  +----------------------------------------+  |
|  |  CMT (Cached Mapping Table)            |  |
|  |  LRU-based, separate read/write caches |  |
|  |  budget: 0.5% of SSD capacity          |  |
|  |  read:write ratio 1:1                  |  |
|  +----------------------------------------+  |
|  |  Translation Page Manager              |  |
|  |  - 16KB pages, frame-based layout      |  |
|  |  - Regular entries: 32B (hash+PPA+meta)|  |
|  |  - Inline entries: variable (multi-    |  |
|  |    frame)                              |  |
|  |  - OOB bitmaps for entry reconstruction|  |
|  |  - Sorted list within each page        |  |
|  |  - [ ML HOOK: inline decision ]        |  |
|  +----------------------------------------+  |
|  |  Data Page Manager                     |  |
|  |  - Variable-size KV pair packing       |  |
|  |  - Sector-aligned storage              |  |
|  |  - Valid/invalid entry tracking         |  |
|  +----------------------------------------+  |
|  |  GC Engine                             |  |
|  |  - Greedy victim block selection       |  |
|  |  - KV-aware valid entry copy           |  |
|  |  - Inline-to-regular conversion during |  |
|  |    compaction (KVPack-D)               |  |
|  +----------------------------------------+  |
|  |  Metrics Collector                     |  |
|  |  - Flash reads (data vs translation)   |  |
|  |  - Flash writes + WAF                  |  |
|  |  - CMT hit/miss ratio                  |  |
|  |  - Inline ratio per epoch              |  |
|  |  - Inline-to-regular conversion rate   |  |
|  |  - Per-request latency (modeled)       |  |
|  +----------------------------------------+  |
+----------------------------------------------+
           |
           v
+----------------------------------------------+
|  Flash Emulation Layer                        |
|  - DRAM-backed page/block array               |
|  - Page size: 16KB, Block: 256 pages (4MB)    |
|  - Per-op latency model:                      |
|    read=45us, write=200us, erase=2ms          |
|  - Read/Write/Erase counters per block        |
+----------------------------------------------+
```

## Core Data Structures

### Flash Geometry (default configuration matching KVPack)

| Parameter | Value |
|-----------|-------|
| SSD Capacity | 64 GB |
| Flash Page Size | 16 KB |
| Block Size | 256 pages = 4 MB |
| Page Read Latency | 45 us |
| Page Write Latency | 200 us |
| Block Erase Latency | 2 ms |

### K2P Mapping Table

| Parameter | Value |
|-----------|-------|
| Mapping Table Size | 4 GB (6.25% of SSD capacity) |
| Max KV Pairs | ~134 million |
| Hash Function | MurmurHash3 |
| Hash Length | 27 bits |
| Regular Entry Size | 32 bytes |
| Translation Page Size | 16 KB (= 1 flash page) |
| Entries per Translation Page | 512 (regular entries) |
| Collision Resolution | Quadratic probing across translation pages |

### Mapping Entry Formats

**Regular entry (32 bytes):**
```
| key_hash (8B) | PPA (4B) | offset (2B) | value_len (2B) | key_len (2B) | reserved (14B) |
```

**Inline entry (variable, multi-frame):**
```
| key_hash (8B) | key_len (2B) | value_len (2B) | value_data (variable) | padding to frame boundary |
```

### Translation Page Layout (per KVPack Fig. 2-3)

```
Main Area (16KB):
  [ Frame 0 ][ Frame 1 ][ Frame 2 ] ... [ Frame N-1 ]
  Each frame = 32 bytes (= regular entry size)
  Inline entries span multiple consecutive frames

OOB / Spare Area:
  N   - number of mapping entries
  S   - number of frames (slabs) used
  B1  - bitmap identifying entry type (regular vs inline)
  B2  - bitmap identifying frame count per entry
```

### CMT Configuration

| Parameter | Value |
|-----------|-------|
| Total Budget | 0.5% of SSD capacity |
| Read CMT : Write CMT | 1:1 ratio |
| Eviction Policy | LRU |
| Inline entries cached? | No (KVPack policy) |
| Working set : CMT ratio | 10:1 (default eval) |

## Workloads

### Primary Evaluation Workloads

The following 10 workloads are used by KVPack (Table V in paper), originally identified by AnyKey (ASPLOS 2025). All use uniform random key distribution.

| Name | Source | Avg Key (B) | Avg Value (B) | KV Pair (B) | Notes |
|------|--------|-------------|---------------|-------------|-------|
| ETC | Facebook | 41 | 358 | 399 | Largest values; tests regular entry path |
| UDB | Facebook | 27 | 127 | 154 | Medium values |
| ZippyDB | Facebook | 48 | 43 | 91 | Small values; strong inline candidate |
| Cache | Twitter | 42 | 188 | 230 | Medium values |
| Cache15 | Twitter | 38 | 38 | 76 | Very small values; strong inline candidate |
| VAR | Facebook | 35 | 115 | 150 | Medium values |
| Crypto1 | BlockStream | 76 | 50 | 126 | Large keys, small values |
| Crypto2 | Trezor | 37 | 110 | 147 | Medium values |
| Dedup | IBM | 20 | 44 | 64 | Small KV pairs; strong inline candidate |
| RTDATA | Microsoft | 24 | 10 | 34 | Smallest values; fits in single frame |

### Twitter Cache Traces

From the [Twitter cache trace dataset](https://github.com/twitter/cache-trace) (Yang et al., OSDI 2020):

- **Cache**: cluster with avg key size 42B, avg value size 188B
- **Cache15**: cluster 15 with avg key size 38B, avg value size 38B

Trace format (CSV, zstd-compressed):
```
timestamp, anonymized_key, key_size, value_size, client_id, operation, TTL
```

Operations: `get`, `gets`, `set`, `add`, `replace`, `cas`, `append`, `prepend`, `delete`, `incr`, `decr`

**Mapping to emulator operations:**
- `get` / `gets` -> `GET(key)`
- `set` / `add` / `replace` -> `PUT(key, value)` (synthetic value of recorded `value_size`)
- `delete` -> `DELETE(key)`
- Other ops (`cas`, `append`, `incr`) -> mapped to `PUT` or skipped

### Synthetic Workloads

For controlled experiments, the emulator also supports synthetic workload generation:
- Configurable key/value size distributions (uniform, Zipfian)
- Configurable read/write ratios
- Mixed workloads (e.g., 80% small KV pairs + 20% large, as used in KVPack CMT evaluation)
- Phase-change workloads for testing dynamic adaptation

## ML Integration

### Hook Point

The primary ML integration point is the **inline decision function**, called during PUT operations:

```python
def should_inline(key: bytes, value: bytes, context: InlineContext) -> bool:
    """
    Decides whether a KV pair should be stored as an inline mapping entry
    or as a regular entry with a separate data page.

    Baseline (hash table):  return len(value) <= PPA_SIZE  (always ~8B)
    KVPack-S:               return len(value) <= static_threshold
    KVPack-D:               return len(value) <= dynamic_threshold(profiler)
    ML policy:              return model.predict(features) > threshold
    """
```

### Context Features Available to ML Models

```python
@dataclass
class InlineContext:
    key_size: int               # bytes
    value_size: int             # bytes
    tp_utilization: float       # fraction of frames used in target translation page
    tp_inline_ratio: float      # fraction of inline entries in target translation page
    cmt_hit_rate: float         # recent CMT hit rate (sliding window)
    cmt_pressure: float         # CMT utilization (entries / capacity)
    access_frequency: float     # estimated access frequency for this key
    epoch: int                  # current profiling epoch
    total_flash_reads: int      # cumulative flash reads
    tp_flash_reads: int         # cumulative translation page reads
    data_flash_reads: int       # cumulative data page reads
```

### Planned ML Policies

1. **Linear Regression**: predict expected flash reads saved by inlining vs. cost of increased translation page size. Inline when net benefit > 0.
2. **Contextual Bandit**: arms = {inline, regular}. Context = feature vector above. Reward = negative flash reads for subsequent accesses to this key.

### Decision Granularity Options

| Granularity | Description | Overhead | Accuracy |
|-------------|-------------|----------|----------|
| Per-request | Call model on every PUT | Highest | Highest |
| Per-translation-page | Set threshold per TP based on its workload profile | Medium | Medium |
| Per-epoch | Batch decision every N operations | Lowest | Coarsest |

## Evaluation Metrics

Matching KVPack's evaluation methodology:

| Metric | Description |
|--------|-------------|
| **Flash Page Reads** | Total, broken down by data page reads vs translation page reads |
| **Flash Page Writes** | Total, broken down by data page writes vs translation page writes |
| **WAF** | Write Amplification Factor = total flash writes / host-requested writes |
| **Read Latency** | Per-request, modeled from flash read count x page_read_latency |
| **CMT Hit Rate** | Fraction of mapping lookups served from DRAM cache |
| **Inline Ratio** | Fraction of mapping entries stored as inline entries |
| **Inline-to-Regular Conversion Rate** | Fraction of inline entries evicted to regular during insertion |
| **Requests with Flash Reads <= 1** | Percentage of read requests completing in at most 1 flash read |

## Project Structure

```
kv-emulator/
  src/
    flash.py            # Flash emulation layer (pages, blocks, read/write/erase)
    mapping.py          # K2P mapping table, GMD, translation pages
    cmt.py              # Cached Mapping Table (LRU, read/write split)
    kvssd.py            # KV-SSD firmware emulator (ties everything together)
    gc.py               # Garbage collection engine
    metrics.py          # Performance metrics collection and reporting
    inlining/
      baseline.py       # TurboHash-style static inlining (value <= PPA size)
      kvpack_s.py       # KVPack-S: static threshold via profiling
      kvpack_d.py       # KVPack-D: dynamic threshold via continuous profiling
      ml_policy.py      # ML-based inlining decision logic
    workloads/
      trace_replay.py   # Twitter/Facebook trace replay engine
      synthetic.py      # Synthetic workload generator
      workload.py       # Workload abstraction and configuration
  tests/
    test_flash.py
    test_mapping.py
    test_cmt.py
    test_kvssd.py
    test_inlining.py
  configs/
    default.yaml        # Default SSD configuration (64GB, 16KB pages, etc.)
  scripts/
    run_evaluation.py   # Run full evaluation suite
    plot_results.py     # Generate comparison plots
  data/
    traces/             # Downloaded trace files (gitignored)
  results/              # Evaluation output (gitignored)
  README.md
  requirements.txt
```

## Configuration

Default configuration mirrors KVPack's evaluation setup:

```yaml
ssd:
  capacity_gb: 64
  dram_ratio: 0.01          # 1% of SSD capacity as integrated DRAM

flash:
  page_size_kb: 16
  pages_per_block: 256      # block = 4MB
  read_latency_us: 45
  write_latency_us: 200
  erase_latency_us: 2000

mapping:
  table_size_gb: 4          # 6.25% of SSD capacity
  hash_function: murmurhash3
  hash_bits: 27
  entry_size_bytes: 32      # regular mapping entry
  max_retry: 8              # quadratic probing bound

cmt:
  budget_ratio: 0.005       # 0.5% of SSD capacity
  read_write_ratio: 1.0     # 1:1
  eviction_policy: lru

inlining:
  mode: kvpack_s            # baseline | kvpack_s | kvpack_d | ml
  profiler_warmup_ios: 1000000  # 1M I/Os before setting threshold
  profiler_query_interval: 100000  # KVPack-D re-query interval

workload:
  queue_depth: 64
  working_set_to_cmt_ratio: 10
  key_distribution: uniform_random
```

## References

- Saha et al., "KVPack: Dynamic Data Inlining in Mapping Table for Key-Value Storage Devices," IEEE Transactions on Computers, 2025.
- Saha et al., "MoKE: Modular Key-value Emulator for Realistic Studies on Emerging Storage Devices," IEEE CLOUD, 2023.
- Park et al., "AnyKey: A Key-Value SSD for All Workload Types," ASPLOS, 2025.
- Yang et al., "A Large-scale Analysis of Hundreds of In-memory Cache Clusters at Twitter," OSDI, 2020.
- Li et al., "The CASE of FEMU: Cheap, Accurate, Scalable and Extensible Flash Emulator," FAST, 2018.
- Samsung, "OpenMPDK KVSSD," https://github.com/OpenMPDK/KVSSD
- Twitter Cache Traces: https://github.com/twitter/cache-trace
