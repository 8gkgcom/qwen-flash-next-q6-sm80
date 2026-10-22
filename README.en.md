# Qwen Flash Next Q6 on SM80

Experimental GGUF Q6_K integration for Qwen3.8 Flash Next on two 64 GiB CMP 170HX GPUs, using a pinned SM80 vLLM fork. A 50.66 GiB Q8_0 PLE table stays compressed in host RAM; only requested rows are decoded into BF16 buffers.

[中文首页](README.md) · [Installation](docs/INSTALL.md) · [Tests](docs/TESTING.md) · [Architecture](docs/ARCHITECTURE.md) · [Results](docs/RESULTS.md) · [Attribution](THIRD_PARTY_NOTICES.md)

**Experimental source release.** The runtime comes from a working native deployment. The newly packaged Docker recipe has not yet completed a fresh build and full-model validation on an idle dual-GPU host. Local CPU checks do not imply full GPU validation.

The preset uses TP1/PP2 with 25/23 layers, 262144 context, two active requests plus four waiting, 1024 batched prefill tokens, a 0.96 memory budget, BF16 state/KV, full decode graphs, MTP2 for one request and MTP1 for two. The KV pool is shared: two full 262K requests cannot be accommodated at once. Text input only. Model plus draft files need approximately 160 GiB of storage, excluding images and caches. The reference host has 96 GB RAM.

Local contributions include the Flash Next GGUF loader, compressed PLE integration, negative-expert padding protection, dynamic-MTP graph compatibility, mixed-prefill scheduling guards, request-ID snapshots, reduced CPU spinning and asynchronous metadata uploads. vLLM, GGUF kernels, MTP, RAM offload and the SM80 foundation are credited to their respective upstream projects; this project makes no first-invention claim.

Historical measurements on the native host: approximately 64.5–83.6 output tokens/s on three 512-token probes at temperature 1 with xhigh thinking; approximately 2.15% improvement from five asynchronous metadata transfers. The probes often end during reasoning and are not complete coding accuracy evaluations. Raising prefill from 1024 to 2048 did not improve decode speed and worsened latency, so 1024 remains the default.

```bash
python3 scripts/fetch_sources.py
docker build -t qwen-flash-sm80:0.2.0 build/deps/sm80
docker build -t flashnext-q6-sm80:0.1.0 .
# Follow docs/INSTALL.md to download weights and prepare /data/runtime.
python3 scripts/serve.py --config config/local.json --dry-run
```

`--start` is explicit and refuses occupied GPUs. No scripts stop existing services, change host drivers or install autostart units. The API is published only on host loopback by default. Credentials and weights are not bundled. New local code is Apache-2.0; inherited code, model weights and external dependencies retain their respective terms.
