# NCCL Collectives on 4× H100 (NVLink / NVSwitch) — Results

Single-node, 4× H100 80GB SXM5, intra-node NVSwitch (all pairs NV18). Driver 580.159.03, NCCL 2.18.3, nccl-tests, `-g 4 -w 5 -n 50`, sizes 8 B → 8 GB.

**NVLink budget (measured):** 18 links × 26.562 GB/s = **478 GB/s** per-GPU unidirectional.

## 1. Bandwidth & latency by collective

| collective | peak busbw | % of NVLink uni | small-msg latency floor | 50%-of-peak crossover |
|---|---|---|---|---|
| all_reduce | 366 GB/s | 77% | 22.7 µs | 8MB |
| all_gather | 344 GB/s | 72% | 16.8 µs | 16MB |
| reduce_scatter | 350 GB/s | 73% | 21.4 µs | 16MB |

Small messages sit on a ~23 µs latency floor (kernel launch + handshake); bandwidth only ramps past ~1–4 MB. This is exactly why TP=N LLM inference is communication-bound at small batch/sequence: each layer's all-reduce moves few bytes and pays the floor, not the bandwidth.

## 2. Algorithm comparison — all_reduce busbw (GB/s)

| algorithm | 256MB | 1GB | 8GB |
|---|---|---|---|
| Ring | 340 | 356 | 366 |
| Tree | 236 | 250 | 259 |
| NVLS | 359 | 372 | 376 |

**NVLS (NVLink SHARP) wins at every size.** NVSwitch performs the reduction in-network via multicast/reduction engines (`NVLS multicast support ... NVLS_NCHANNELS 16`), so each GPU ships its data once instead of the 2(N−1)/N passes a ring needs. Tree trails badly on a single node — it is built for multi-node latency, not intra-node bandwidth. CollnetChain needs IB SHARP (multi-node) and is correctly unavailable here.

## 3. Protocol comparison — all_reduce @256 MB busbw (GB/s)

| protocol | busbw | note |
|---|---|---|
| Simple | 340 GB/s | max bandwidth, large messages |
| LL128 | 313 GB/s | latency/bandwidth compromise |
| LL | 147 GB/s | low-latency, tiny messages only (no fences) |

Confirms the protocol ladder: **LL** sacrifices ~57% bandwidth to drop latency for small payloads; **Simple** is the bandwidth play for big transfers; **LL128** sits between. NCCL's autotuner switches protocol by message size — these are the endpoints it interpolates.

## 4. Topology

All GPU pairs report **NV18** (18 bonded NVLink-4 → full NVSwitch fabric):

```
GPU0	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NV18	PXB	PXB	NODE	NODE	SYS	SYS	SYS	SYS	NODE	NODE	SYS	SYS	0-31,64-95	0		N/A
GPU1	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NV18	PXB	PXB	NODE	NODE	SYS	SYS	SYS	SYS	NODE	NODE	SYS	SYS	0-31,64-95	0		N/A
GPU2	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NODE	NODE	PXB	PXB	SYS	SYS	SYS	SYS	NODE	NODE	SYS	SYS	0-31,64-95	0		N/A
GPU3	NV18	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NODE	NODE	PXB	PXB	SYS	SYS	SYS	SYS	NODE	NODE	SYS	SYS	0-31,64-95	0		N/A
GPU4	NV18	NV18	NV18	NV18	 X 	NV18	NV18	NV18	SYS	SYS	SYS	SYS	PXB	PXB	NODE	NODE	SYS	SYS	NODE	NODE	32-63,96-127	1		N/A
GPU5	NV18	NV18	NV18	NV18	NV18	 X 	NV18	NV18	SYS	SYS	SYS	SYS	PXB	PXB	NODE	NODE	SYS	SYS	NODE	NODE	32-63,96-127	1		N/A
GPU6	NV18	NV18	NV18	NV18	NV18	NV18	 X 	NV18	SYS	SYS	SYS	SYS	NODE	NODE	PXB	PXB	SYS	SYS	NODE	NODE	32-63,96-127	1		N/A
GPU7	NV18	NV18	NV18	NV18	NV18	NV18	NV18	 X 	SYS	SYS	SYS	SYS	NODE	NODE	PXB	PXB	SYS	SYS	NODE	NODE	32-63,96-127	1		N/A
```

## 5. What this implies for LLM serving

- **TP=4 all-reduce** runs once per transformer layer (after attention and after MLP). At decode batch=1 each all-reduce is a few hundred KB — below the crossover, so latency-bound. Throughput-serving (large batch) pushes into the bandwidth regime where NVLS's ~376 GB/s matters.

- **Choosing NVLS** (`NCCL_ALGO=NVLS`, default-on for NVSwitch) buys ~3–8% over Ring at the sizes TP inference uses, for free.

- The measured ceiling (~77% of NVLink uni budget for ring all-reduce) is the honest number to plan around — not the 478 GB/s datasheet figure.

