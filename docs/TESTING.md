# 测试、复现与验收

分清三个层次：独立 CPU 逻辑检查、固定框架内的数值/兼容检查、整模型 API 验收。仓库 CI 仅执行第一层；GPU 测试必须在空闲设备上主动执行。整理发布期间不向原模型服务发送压测，不停止原服务。

## 1. 独立 CPU 检查

Linux / WSL，Python 3.12 和 g++，不需要模型、PyTorch 或 GPU：

```bash
PYTHONPATH=src Q6_REQUIRE_CPP=1 python3 -m unittest discover -s tests -v
python3 scripts/audit_release.py
```

覆盖 FIFO 的 2 运行 / 4 等待、溢出、流式租约、排队断连和取消释放，预填充/生成门控的交替与异常恢复，部署配置边界、GPU 占用拒绝、密钥不进入命令、覆盖层先校验后安装、归档路径穿越拒绝，以及 C++ Q8_0 行解码。

原生解码器测试用独立的 Python IEEE FP16 解析和 BF16 舍入作为参考，穷举 65,536 种 FP16 缩放编码，覆盖 NaN/Inf、正负量化字节、重复/重排索引、越界不改写输出和空输入。部署配置采用 Linux 路径，Windows 用户请通过 WSL 执行整套检查；原生 Windows 不支持本包的部署和 Linux 动态库验证。

## 2. 固定镜像内的环境检查

先完成镜像构建，在空闲测试机执行。这两项运行固定 vLLM 的 Python 路径，不启动模型；暴露 GPU 是为了提供框架所需的 CUDA 驱动环境：

```bash
docker run --rm --gpus all --entrypoint python3 flashnext-q6-sm80:0.1.0 \
  tests/runtime/check_dynamic_graphs.py
docker run --rm --gpus all --shm-size=1g --entrypoint python3 flashnext-q6-sm80:0.1.0 \
  tests/runtime/check_cpu_ipc.py
```

动态图检查必须先复现原候选形状错误，再确认草稿的单 token 候选与目标验证候选分别正确。IPC 检查发送 200 条消息，故意丢弃写通知，检查顺序、超时和取消。

有模型及已准备的加载视图时，增加实际行查表和 PLE 哈希校验：

```bash
docker run --rm --gpus all --mount "type=bind,source=$Q6_DATA,target=/data,readonly" \
  --mount "type=bind,source=$Q6_MODELS,target=/data/models,readonly" \
  --entrypoint python3 flashnext-q6-sm80:0.1.0 tests/runtime/check_real_q8_rows.py
docker run --rm --gpus all --mount "type=bind,source=$Q6_DATA,target=/data,readonly" \
  --mount "type=bind,source=$Q6_MODELS,target=/data/models,readonly" \
  --entrypoint python3 flashnext-q6-sm80:0.1.0 tests/runtime/check_ple_hashing.py
```

它们使用只读映射而不 mlock 整表。实际行测试比较 1026 行及全部缩放编码；PLE 测试涵盖历史携带、EOS 重置、2/3-gram、精确 int64 常量和补齐 token 排除。结果 JSON 写在容器临时目录并打印摘要。

## 3. 空闲 GPU 上的数值检查

这些检查会占用 GPU。默认拒绝运行，需要设置 `Q6_RUN_GPU_TESTS=1`，且检测到任何 GPU 计算进程时退出。**在启动整模型之前执行**：

```bash
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 -e Q6_RUN_GPU_TESTS=1 \
  --mount "type=bind,source=$Q6_DATA,target=/data,readonly" \
  --mount "type=bind,source=$Q6_MODELS,target=/data/models,readonly" \
  --entrypoint python3 flashnext-q6-sm80:0.1.0 tests/runtime/test_gpu_ops.py

docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 -e Q6_RUN_GPU_TESTS=1 \
  --mount "type=bind,source=$Q6_DATA,target=/data,readonly" \
  --mount "type=bind,source=$Q6_MODELS,target=/data/models,readonly" \
  --entrypoint python3 flashnext-q6-sm80:0.1.0 tests/runtime/check_gpu_padding.py

docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 -e Q6_RUN_GPU_TESTS=1 \
  --entrypoint python3 flashnext-q6-sm80:0.1.0 tests/runtime/check_shortconv_async.py
```

第一项以真实小段 Q6 权重验证线性层和 MoE；误差阈值保留历史验证的相对 L2 界限（线性层 0.04、MoE 0.08），不是整模型精度保证。第二项验证负专家槽位输出为零、变化输入的 CUDA Graph 回放。第三项比较 24 组 CPU/GPU 元数据构造及 256 个异步上传源的生命周期。

`test_gpu_ops.py` 为排错启用 CUDA 同步执行；不要继承这个环境去测性能。新 Docker 包的这些检查在发布整理时仅做静态整理，没有重新执行 GPU 验收。

## 4. 整模型 API 与容量

按安装文档启动隔离服务，再执行：

```bash
python3 scripts/check_api.py --base-url http://127.0.0.1:18421
```

检查健康状态、模型别名、两个同时提交的简单算术请求，以及请求完成后队列释放。它是基本功能检查，不证明复杂题正确率。

32K 与完整窗口检索：预填充脚本需要客户端环境有 httpx（基础镜像已有，也可以在测试虚拟环境安装）。

```bash
python3 scripts/benchmark_prefill.py --base-url http://127.0.0.1:18421 \
  --label baseline-32k --input-tokens 32768 --mixed --output runs/32k.json
python3 scripts/benchmark_prefill.py --base-url http://127.0.0.1:18421 \
  --label baseline-262k --input-tokens 262000 --mixed --output runs/262k.json
```

脚本使用服务的 tokenizer 准备固定长度输入，每次请求采用新的 cache salt，记录首 token 和长请求期间短请求的总时间。温度 0、关闭思考，检查首尾标记能否正确回取。完整窗口测试是单长请求加短请求；不能同时提交两个完整 262K 请求。

进一步人工验收：流式断开后再次请求、队列满返回 429、工具调用与工具结果回传、两条较长请求并发、模型关闭后的资源释放。出现 OOM、CUDA 错误、请求挂住或输出异常就停止该候选试验，保留基线配置；不要自动修改驱动或让测试脚本替你重启原服务。

## 5. 单请求速度 A/B

保留同样的硬件、模型文件、上下文、并发上限、GPU 顺序、采样、提示词和输出上限；只改一个优化开关。先确保无其它客户端请求，也没有挖矿等竞争负载。每次启动后有一次预热，然后三题各重复两次：

```bash
python3 scripts/benchmark.py --base-url http://127.0.0.1:18421 \
  --label baseline --output runs/baseline.json
# 更换一个候选配置并在空闲测试服务中重启后：
python3 scripts/benchmark.py --base-url http://127.0.0.1:18421 \
  --label candidate --output runs/candidate.json
python3 scripts/compare_benchmarks.py runs/baseline.json runs/candidate.json
```

默认 temperature=1、top-p=0.95、top-k=20、presence penalty=0.5、seed=42、xhigh，固定 512 tokens。默认只保存计时、usage 和输出哈希；`--save-text` 才把生成文本写入个人的 `runs/`。请求中显式采样参数用于减少服务默认值漂移。

对照工具拒绝模型别名、采样、提示词或度量方式不同的文件。配置/硬件、日志错误和显存余量仍需人工记录；工具不会从两个结果文件自动判断所有环境条件相同。建议做 baseline → candidate → baseline → candidate 的多轮交替，改善预热、温度和时钟偏差。

性能接受条件：没有数值/功能回归，单请求生成有可重复收益，长输入与混合请求延迟可接受，所需上下文与并发压力测试通过。单次快一点或仅能启动都不够。

## 6. 代码能力测试

性能探针不自动运行模型生成的代码，512 tokens 也通常不足以得到完整答案。若比较温度或模型能力，使用 [代码质量验收题](../benchmarks/coding-quality.md)，增加输出上限，在隔离环境执行人工审阅后的测试，记录完成率、测试通过率和完成一个正确任务的总时间。不要把速度探针的思考截断当作模型答错。
