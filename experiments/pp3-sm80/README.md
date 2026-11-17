# 三卡 Q6 / MTP2 实验补丁

2026-11-17，从已运行的三卡部署提取。保留 Q6_K 主权重、固定 MTP2、4 运行 / 4 等待，以及 18/14/16 分层。中间一张风冷卡减少层数。

这是供审阅和移植的源码补丁，**不是新的开箱即用安装包**。仓库根目录仍是原来的双卡方案；原 `scripts/serve.py` 不支持直接读取本目录的三卡配置。三卡测试使用另行适配的 SM80 v0.3.0 环境，发布整理后的独立构建和 CPU 缓存管理检查已通过；全新机器整模型启动尚未验收。不要把旧双卡 Docker 的验证结果算到本目录。

## 改了什么

- `q6_speed.py`：只对草稿词表使用 INT4 Marlin 初筛，再用原 Q8 权重重算 top-32；最终仍由 Q6 目标模型验证。主模型权重不重新量化。筛选不是全词表精确 argmax 的数学保证。
- `q6_moe_speed.cu`：解码专家算子每个线程块处理两行。已测的三层、全部 512 专家与原向量算子逐元素一致。
- `moe.cuh`：修正批量算子硬编码的 255 专家编号上限。包装器只接受 512 专家；同时修正补齐块边界判断。批量与向量路径仍有正常浮点累加差异。
- `q6_ram_kv.py`：以固定槽数、分卡私有 pinned RAM 缓冲保存完整的 3456-token 检查点；24 GiB 总预算实际约 19.8 GiB。缓存池满后按 LRU 淘汰未占用缓存，保护在途读写。8-token QSA scratch ring 不转移，未完成尾部重新计算。
- 同模块修复 PP 延迟采样的请求槽位表，抢占时丢弃旧的在途采样结果，并限制同时预填充以避免混合批次错误。解码仍支持四路。
- `loader-pp3.patch`：允许 PP3，保留缺失参数和专家分片校验。

额外线性层分块方案虽然微测试更快，但整机每秒解码步数没有稳定改善，未纳入此默认源码。

## 实测结果

三张 CMP 170HX 64 GiB，约 157 GiB 可用系统内存，180 W 功耗上限；测速时暂停挖矿。temperature=0、seed=1234、关闭思考、每题固定生成 1024 tokens。速度按 SSE 首个到最后一个非空内容增量估算，不含首字等待，MTP 多 token 合并发送会产生测量误差。

| 场景 | 原版 tok/s | 优化后 tok/s |
|---|---:|---:|
| 英文教程 | 91.70 | 107.12 |
| 中文教程 | 79.56 | 93.90 |
| 代码生成 | 96.67 | 114.10 |
| 8843-token 输入后生成 | 86.60 | 103.07 |

**没有稳定达到 120 tok/s。** 三个短输入测试输出与原版一致；长输入因专家上限修复等运行变化，输出不同，不能当成完全同输出性能对照。这些生成探针不是代码正确率评测，也不能与仓库历史 temperature=1 / xhigh 结果直接比较。

最终配置完成 8 个请求（4 运行 + 4 等待，第 9 个返回 429），以及四路各约 15K 输入 / 4K 输出，全部编号连续、首尾标记正确。原生环境的 CPU KV 恢复测试通过；此前小显存压力测试验证过抢占、恢复和 LRU 淘汰。本轮最终满容量测试没有触发抢占，不能声称重新覆盖了该路径。完整四路 262K 和长期稳定性尚未验证。见 `results.json`。

## 移植与构建

1. 在隔离目录准备兼容的 vLLM / GGUF / SM80 环境。参考 `runtime-fingerprints.json`；这份三卡环境由 SM80 revision `4d229f1438486fa759727f128fc28f3609423dbb` 移植，并含额外兼容改动。仅 checkout 该 revision 不足以还原环境。本目录不自动安装或覆盖 vLLM。
2. 使用该环境的 Python、PyTorch 和匹配的 CUDA toolkit 运行 `python build.py`，生成本目录 `build/q6_speed_cuda.so`。测试机器使用 CUDA 13.0、SM80；避免混用 pip CUDA 头文件和另一版本 nvcc。
3. 把本目录加入 `PYTHONPATH`，在每个 worker 的 GGUF 插件注册完成后调用 `from q6_speed import install; install()`。现有插件可用 `Q6_SPEED_ENABLE=1` 条件调用；这个环境变量本身不会自动导入模块。旧 `Q6_DRAFT_Q8_MARLIN` 和动态 MTP2/1 路径关闭。草稿模块严格检查 PP3、TP1、MTP2 和 Q8 词表布局。
4. 审阅 `loader-pp3.patch`，再移植到对应版本。`profile.example.json` 列出参数、环境和 KV connector 配置。模型、MTP、tokenizer、GGUF manifest 和本地路径由部署者提供；没有上传权重或私有控制台启动器。
5. 开启异步调度，保持 18/14/16 分层，1024 预填充和 4 路上限。API 4+4 队列需接入已有 admission middleware，vLLM 的 `max_num_seqs=4` 本身不限制外部等待队列。`Q6_KV_CACHE_MAX_BYTES` 是原生适配层开关，不是通用 vLLM 参数；移植时需实现同等的每卡 8 GiB KV 预算。
6. RAM connector 需要匹配的 offloading、scheduler 和 V2 runner API；几何不匹配会拒绝运行。发布版删除了私有诊断接口，日志默认关闭，主动设置 `Q6_RAM_KV_EVIDENCE` 才写入去请求标识的事件。该模块不可直接当成任意模型的显存溢出插件。

先做构建和数值检查，再做完整服务回归。此说明不运行服务、不关闭现有任务、不更改驱动或 BIOS。

## 复现检查

```bash
python build.py
# 只在选定的测试 GPU 空闲时执行；manifest 格式同 prepare_model.py 的输出。
export Q6_GGUF_MANIFEST=/path/to/gguf-manifest.json
CUDA_VISIBLE_DEVICES=0 python test_moe.py
python test_host_manager.py
# 服务由部署者启动；密钥只从环境读取，结果文件不保存密钥或模型输出。
export Q6_API_BASE=http://127.0.0.1:8000
export Q6_MODEL_NAME=your-served-model-name
python benchmark_single.py
python audit_package.py
```

`test_moe.py` 会读取第 0、17、47 层真实 GGUF 专家权重，覆盖 512 专家、融合 gate/up 及 down 路径。运行前确保测试 GPU 有足够空闲显存。`test_host_manager.py` 需要兼容的运行时，可在 CPU 环境测试 LRU 和在途缓存保护。发布整理期间完成了无 GPU 的语法、源码完整性、去敏检查、CUDA 扩展编译及 CPU 缓存管理测试；前述整机数据来自发布前的原生服务。

## P2P 结果与边界

当前主板/驱动组合下，六个方向、四种大小的 `cudaMemcpyPeerAsync` 数据校验：关闭 P2P 为 24/24 通过，开启为 0/24。强制 NCCL P2P 的 CUMEM 和 legacy 两条路径都超时；SHM 路径通过，16–64 KiB 环形传输约 64–91 微秒。它们不能证明所有魔改驱动的 P2P 都无效，也不能把通信内核等待时间全部算作传输瓶颈。默认保持已验证的 SHM。见 `p2p-results.json`；不包含驱动二进制、固件或刷写工具。

## 去敏与来源

只发布允许列表中的源码和汇总数字。未包含主机地址、账号密码、API key、GPU UUID、钱包、BIOS、模型权重、服务配置、进程标识、原始日志或生成文本。文件路径改为包内相对路径或由使用者提供；保留通用环回地址作为示例。

代码继承本仓库 Apache-2.0；CUDA 数学来自 GGUF plugin / llama.cpp，保留其 Apache-2.0 / MIT 声明。见 `NOTICE`、`LICENSE.gguf-plugin`、`LICENSE.llama.cpp`。草稿筛选参考 Qwen-Flash-SM80-170HX 的筛选后原权重复核设计。
