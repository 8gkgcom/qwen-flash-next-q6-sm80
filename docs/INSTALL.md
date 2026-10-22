# 安装与部署

从项目根目录在 Linux x86_64 主机执行。安装目标是独立 Docker 镜像和独立数据目录。已有模型正在运行时，可以阅读代码、下载文件；镜像编译会消耗 CPU、内存和磁盘带宽，建议也安排在空闲时段。不要在仍占用同一组 GPU 的情况下启动第二份模型。

## 1. 条件与版本

- 双 SM80 CMP 170HX，每张可见约 65536 MiB 显存；参考主机 96 GB RAM。
- 启动前至少约 60 GiB 可用主机内存，以容纳压缩 PLE 表和基本余量。更充裕的 RAM 有利于加载过程；mlock 失败会直接报错。
- CUDA 13 兼容的 NVIDIA 驱动、Docker、NVIDIA Container Toolkit、Python 3.12。驱动和 GPU 显存配置由部署者预先准备，本项目不改驱动。
- 模型 + MTP 约 160 GiB，建议为数据至少留 220 GiB，再额外安排 Docker 镜像和编译空间。
- 下载 GGUF 仓库需要 Hugging Face 对该仓库的访问授权。

基础镜像固定为：

```text
vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
```

SM80 源码固定 `11421471c050cccc036b004e91345583e33993b2`，GGUF 插件固定 `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`，gguf-py 固定 llama.cpp `98fd535f104720d35eed43d127dd83013aafabed`。完整 URL 和归档哈希在 [dependencies.lock.json](../dependencies.lock.json)。不要用 `pip install -U vllm` 替换这个组合。

上游基础源码构建包含其 GDS 依赖；本项目运行时关闭 GDS，用 CPU Q8_0 PLE。P2P 和 GDS 都不是本方案已验证的必要加速路径。

## 2. 下载源码并构建独立镜像

```bash
python3 scripts/fetch_sources.py
docker build -t qwen-flash-sm80:0.2.0 build/deps/sm80
docker build -t flashnext-q6-sm80:0.1.0 .
```

源码下载先验证 SHA-256；安装覆盖层先验证所有待覆盖文件，再写入镜像。版本不匹配时停止，不猜测修补。保留基础镜像内的 PyTorch、Triton 和 CUDA 组合；GGUF 扩展用 `TORCH_CUDA_ARCH_LIST=8.0` 编译，Q8 行解码器用普通 g++ 编译。

上游包已存在时下载工具复用其标记，因此不要手工修改 `build/deps/` 后继续称作原始源码；重新检查时使用新的 `--output` 目录。本项目包内的覆盖层还会检查关键输入文件的内容哈希。

该整理版没有在本次发布期间执行全新 Docker 构建或整模型启动；首次构建和 GPU 验收应按 [测试说明](TESTING.md) 完成。

## 3. 在独立数据目录下载模型

以下 `Q6_DATA` 必须位于有足够空间的磁盘，例如你的 NVMe 挂载点。不要改写现有模型文件夹。

```bash
export Q6_DATA="$HOME/flashnext-q6-data"
export Q6_MODELS="$Q6_DATA/models"
mkdir -p "$Q6_DATA/models" "$Q6_DATA/metadata" "$Q6_DATA/cache"
python3 -m venv .venv-hf
.venv-hf/bin/pip install 'huggingface_hub==1.31.0'
.venv-hf/bin/hf auth login
.venv-hf/bin/hf download orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF \
  --revision d366e279688c1f2fd6e75e2f98475d38db534f1d \
  --include 'Qwen3.8-Flash-Next-Uncensored-Q6_K-*.gguf' \
            'Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf' \
  --local-dir "$Q6_MODELS" --max-workers 2
```

Token 由 Hugging Face 登录工具存储在用户缓存中，不写进源码、Dockerfile、镜像或示例 JSON。访问被拒绝时，先在模型网页完成条款/授权，再检查账号和 Token 的仓库权限。

已有完整的相同版本文件可以复用：将 `Q6_MODELS` 设为已有 GGUF 文件目录，跳过重新下载，并在后面的本地配置设置 `model_dir`。独立数据目录只保存加载视图和缓存；模型目录以只读方式挂载，不复制 160 GiB 权重。目录需要五个主模型分片和一个 MTP 文件；不需要 mmproj，因为此适配仅提供文本推理。不要混入其它量化等级或其它版本的分片。

## 4. 准备加载视图

此步骤不传入 GPU；只读 GGUF 头与张量范围，生成元数据，不展开整张 PLE、不重新量化权重。

```bash
docker run --rm \
  --mount "type=bind,source=$Q6_DATA,target=/data" \
  --mount "type=bind,source=$Q6_MODELS,target=/data/models,readonly" \
  --entrypoint python3 flashnext-q6-sm80:0.1.0 \
  scripts/prepare_model.py --download-metadata
```

公开配置、tokenizer 和模板从固定版本的 `Qwen/Qwen3.8-Flash-Next` 下载，验证内容哈希。其中 tokenizer、tokenizer_config、vocab 和聊天模板与原部署逐字节一致；配置只提取文本部分并按原部署设置模型类别、独立词嵌入和兼容字段。

输出为 `$Q6_DATA/runtime/`，内含清单、精确 int64 哈希常量和 `model-config/`。清单里的路径是容器内 `/data/models/...`，因此后续运行必须继续挂载到 `/data`。输出已存在则拒绝覆盖；需要重做时使用新的数据目录。头部和范围检查不能替代对全部权重内容的完整哈希校验；下载器负责固定仓库版本的文件获取。

## 5. 生成个人配置，先检查命令

```bash
cp -n config/example.json config/local.json
python3 - <<'PY'
import json, os
from pathlib import Path
p=Path('config/local.json')
c=json.loads(p.read_text())
c['data_dir']=str(Path(os.environ['Q6_DATA']).resolve())
c['model_dir']=str(Path(os.environ['Q6_MODELS']).resolve())
p.write_text(json.dumps(c,indent=2)+'\n')
PY
python3 scripts/serve.py --config config/local.json --dry-run
```

`gpu_ids` 表示 GPU 的顺序，默认 `["0","1"]`。第一张承载 25 个主模型层，第二张承载 23 层及 MTP。不要照搬其他服务器的 GPU UUID。

采样默认保留原测试设置：温度 1、xhigh、presence penalty 0.5。后者与基础模型官方思考推荐的 0 不同；若要比较，单独做正确率测试，不与性能改动混在一次实验里。

可选 API 密钥：在项目目录之外创建仅自己可读的文件，在 `config/local.json` 设置 `api_key_file` 为其绝对路径。启动器只传文件挂载路径，容器从文件读取到 `VLLM_API_KEY`；不会把值打印到启动命令中。默认端口仅绑定主机回环地址，远程接入可另设 SSH 隧道。不要将个人配置或密钥提交到 Git。

## 6. 空闲 GPU 上启动与检查

```bash
python3 scripts/serve.py --config config/local.json --start
```

这是前台进程，适合先观察加载和报错。也可以在自己创建的 screen/tmux 会话中执行。启动器不操作已有服务，GPU 已被占用时会拒绝。它只做启动前检查，不是跨项目的资源调度器；测试期间应避免其它管理器同时自动拉起模型。

另一个终端检查：

```bash
curl --fail http://127.0.0.1:18421/health
python3 scripts/check_api.py --base-url http://127.0.0.1:18421
python3 scripts/benchmark.py --base-url http://127.0.0.1:18421 \
  --label baseline --output runs/baseline.json
```

配置了密钥时，测试脚本从 `Q6_API_KEY` 环境变量读取；不要将真实密钥写进可发布命令。API 健康响应 200 后才算就绪。首次加载、编译和图捕获可能明显慢于后续启动。

停止仅由本项目启动的容器：

```bash
docker stop flashnext-q6-sm80
```

启动器不自动删除已停止的容器。保留日志后，可以给下一次试验使用新的 `name`；也可由你明确删除已停止的本项目容器。镜像和源码的回退不会触碰原模型部署，原部署的服务由其原管理方式继续控制。

## 常见问题

| 情况 | 处理 |
|---|---|
| 覆盖层哈希不匹配 | 检查固定的基础镜像和源码版本；不要强制覆盖 |
| `mlock` 失败 | 检查主机可用 RAM、容器 memlock 上限及 IPC_LOCK；不自动退回硬盘换页 |
| 缺少 `q8_rows.so` / GGUF 扩展 | 检查镜像构建日志、g++、CUDA 13 和 SM80 编译设置 |
| GPU 已占用 | 保持原服务运行；安排空闲机器或明确的维护窗口再试 |
| 长输入 OOM | 保持 0.96 / 1024；两请求共享 KV，不支持同时两个完整 262K |
| GPU 利用率高但速度不高 | 利用率可能包含通信等待；同时记录 TTFT、输出 tokens/s、请求总耗时 |
| 开了 P2P 仍走 SHM | 以 NCCL 实际传输日志为准，不能由配置项推断 P2P 已生效 |
