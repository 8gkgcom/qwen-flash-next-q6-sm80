# Qwen Flash Next Q6 on SM80

面向双 CMP 170HX 的 **Qwen3.8 Flash Next GGUF Q6_K / vLLM** 实验性部署项目。
主模型在两张 GPU 上运行，约 **50.66 GiB 的 Q8_0 PLE 表驻留系统内存，仅解码当前需要的行**，再将 BF16 行结果交给 GPU。

[安装部署](docs/INSTALL.md) · [测试与复现](docs/TESTING.md) · [实现与限制](docs/ARCHITECTURE.md) · [实测记录](docs/RESULTS.md) · [上游与许可](THIRD_PARTY_NOTICES.md) · [English](README.en.md)

**状态：实验性源码发布。** 核心代码来自已经运行并完成多轮测试的原生部署；发布整理在独立目录完成，未重启原服务。新整理的 Docker 安装流程尚未在空闲双卡机器上完成从零构建和整模型验收。CPU 检查可以独立运行，不能替代 GPU 验收。

## 解决的问题

- 将该 GGUF 模型的 HC、PLE、QSA、GDN 和独立 MTP 权重映射到固定版本的 SM80 vLLM。
- 用压缩的 Q8_0 PLE 行表代替约 95.4 GiB 的整表 BF16 展开，保持大表在主机内存，避免逐行从硬盘换页。
- 修复此版本组合中的 MoE 空专家槽位、动态 MTP 图形状、异步请求 ID 快照和混合预填充兼容问题。
- 保留单请求 MTP2 / 双请求 MTP1、完整解码 CUDA Graph、两运行请求和四等待请求。
- 提供 CPU 通知等待、草稿 Q8 词表筛选、五处 PLE 调度索引异步上传，以及可单独关闭的开关。

这是在已有框架上的工程适配。vLLM、GGUF 内核、RAM PLE offload 和 MTP 等基础能力来自上游；本项目不宣称发明这些技术。

## 固定参数

| 项目 | 默认值 |
|---|---|
| GPU / 主机内存 | 2 × 64 GiB CMP 170HX，参考机器 96 GB RAM |
| 模型 | OrcaRouter Qwen3.8 Flash Next Uncensored GGUF Q6_K + 独立 MTP |
| 并行 | TP=1、PP=2，主模型层数 25/23 |
| 上下文 | 262144，包含输入与输出 |
| 并发 / 等待 | 2 / 4；两个请求共享 KV 总池 |
| 预填充总预算 / 长请求阈值 | 1024 / 768 tokens |
| 显存预算 | 0.96 |
| 计算 / KV | BF16 / auto；GGUF 算子的激活量化见实现说明 |
| MTP | 一请求 2 草稿 token，两请求各 1 |
| 图模式 | `FULL_DECODE_ONLY`，关闭 `torch.compile` 图编译模式 |
| 前缀缓存 / 分块预填充 | 开启 |
| 思考 / 采样 | xhigh、temperature 1、top-p 0.95、top-k 20 |
| presence penalty | 0.5，沿用实测配置；基础模型官方思考建议为 0 |
| 接口 | OpenAI 风格 API，默认仅发布到本机 `127.0.0.1:18421` |

2 并发不表示两个请求能同时使用完整的 262K 窗口。仅验证文本输入；不提供视觉、RAM KV offload 或其它 GPU 型号的支持承诺。模型及 MTP 文件合计约 160 GiB，另需镜像、编译和缓存空间。

## 开始使用

先阅读 [安装部署](docs/INSTALL.md)。所有命令在 Linux 部署机执行，源目录与现有服务分开。

```bash
# 下载并校验固定的上游源码；不启动 GPU 或修改系统服务。
python3 scripts/fetch_sources.py

# 先构建固定 SM80 基础，再构建 Q6 适配镜像。
docker build -t qwen-flash-sm80:0.2.0 build/deps/sm80
docker build -t flashnext-q6-sm80:0.1.0 .
```

下载模型、准备加载视图和本地配置后，先查看启动计划：

```bash
python3 scripts/serve.py --config config/local.json --dry-run
# 仅在所选 GPU 空闲、准备好测试时执行：
python3 scripts/serve.py --config config/local.json --start
```

启动器检查 GPU 占用和端口，遇到已有计算进程会退出。它不会停止其它容器、杀进程、修改驱动、注册开机服务或改写已有控制台。

## 有限范围的历史结果

测试机器：i5-12500T、96 GB RAM、双 CMP 170HX 64 GiB；日期 2026-10-22。以下结果来自原生部署，不代表新 Docker 安装已通过整模型验收。

| 优化或对照 | 观察结果 |
|---|---|
| 调度进程改用通知等待 | CPU 占用约 96.8% → 6.4%，100% 对应一个逻辑核心；生成速度基本不变 |
| 五处索引异步 H2D | 三道短输入题的生成速度平均相对提高约 2.15%；12 次输出哈希一致 |
| 当前短输入单请求探针 | 约 64.5–83.6 tokens/s，temperature=1、xhigh、固定生成 512 tokens |
| 预填充 1024 → 2048 | 单请求生成速度无提升；32K 预填充首 token 变慢约 2.55% |
| 2048 的混合请求延迟 | 长请求期间的短请求约 4.02 → 10.64 秒，因此恢复 1024 |

512-token 探针可能全部处于思考阶段，**不是完整代码正确率测试**。首 token、后续生成、端到端耗时和并发吞吐需分别测量；历史温度 0 的短题不能与这里的思考速度混为一谈。[结果与负面实验](docs/RESULTS.md)

## 仓库结构

```text
src/flashnext_gguf/       Q6 加载、Q8 行解码、运行优化和 API 排队
runtime/                 固定 SM80 与 GGUF 插件的增量源文件
patches/                 便于审阅的上游差异
scripts/                 下载源码、部署、测试、审计、打包
tests/                   CPU 回归；runtime/ 为显式执行的环境/GPU 检查
benchmarks/              三道测试题与去标识的历史测量数据
provenance/              文件来源、安装前后 SHA-256、验证范围
docs/                    安装、测试、实现、结果和发布说明
licenses/                保留的第三方许可声明
```

仓库不包含模型权重、Token、账号密码、GPU UUID、主机地址、私有控制台、原始服务器日志或编译二进制。生成的数据、缓存、个人配置和测试输出留在 Git 忽略目录。

## 来源与许可

主要基础为 [Qwen-Flash-SM80-170HX v0.2.0](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/tree/11421471c050cccc036b004e91345583e33993b2)、[vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin/tree/e2b8ad532b8b5ea175100202c30430c1d2b5e6a8) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/98fd535f104720d35eed43d127dd83013aafabed)。
本地新增代码采用 Apache-2.0；第三方部分保留原许可。模型及 CUDA 等依赖独立适用其条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
