# 上游来源与许可

本项目新增的部署、测试、适配辅助代码采用 [Apache-2.0](LICENSE)。继承、修改或依赖的第三方内容保留其各自声明，模型和工具链不因本项目而重新授权。

| 组件 | 固定来源 | 本项目使用方式 / 许可 |
|---|---|---|
| SM80 推理方案 | [Qwen-Flash-SM80-170HX](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/tree/11421471c050cccc036b004e91345583e33993b2)，v0.2.0 | 固定基础，继承 SM80/vLLM 模型与内核集成；[Apache-2.0](licenses/SM80-LICENSE)、[原 NOTICE](licenses/SM80-NOTICE) |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm)，实际基线为锁定的 `qwen38-flash-next` 镜像 | 模型、流水并行、调度、CPU PLE、动态图、推测解码；[Apache-2.0](licenses/vLLM-LICENSE)，文件头保留 |
| 社区 PLE 集成 | 上游注明的 qwen3.8-Flash-DGX / blazux 工作 | 继承的 RAM PLE 基础；[原许可](licenses/PLE-community-LICENSE) |
| GGUF plugin | [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin/tree/e2b8ad532b8b5ea175100202c30430c1d2b5e6a8) | Qwen3.5 GDN 布局工具、GGUF 量化/加载、GPU 算子；[Apache-2.0](licenses/GGUF-plugin-LICENSE) |
| llama.cpp / gguf-py | [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/98fd535f104720d35eed43d127dd83013aafabed) | GGUF 格式、Q8_0 定义、读取/反量化、Flash Next 转换命名规则；[MIT](licenses/llama.cpp-LICENSE) |
| PyTorch、Triton、Marlin、FlashInfer 等 | 随固定镜像/上游依赖提供 | 计算、编译、通信和量化内核；保留各自包内 LICENSE / NOTICE，不将其计作本地发明 |

vLLM 镜像内版本文件报告 `0.1.dev20073+g8e685d198`。这不是一个在本项目中承诺可从 vLLM 主仓库直接检出的正式 tag；可复现基线由完整镜像 digest 与覆盖层 SHA-256 确定。不要仅依赖宿主 Python 的 `pip show vllm` 版本号判断兼容性。

## 本地修改的归属

- `src/flashnext_gguf/`：本地 Q6 加载、PLE 按行解码集成和独立 API 队列；适配规则明确参考上述模型和 GGUF 实现。
- `src/flashnext_gguf/optimizations/`：本地动态 MTP、混合预填充、CPU 通知、Q8 草稿头、异步元数据传输。
- `runtime/overlay/`：在固定 SM80/vLLM 文件上的本地兼容改动，保留上游头部；新增 PP 草稿 scatter / RAM hooks 单独标注来源。
- `runtime/plugin-overlay/`：在固定 GGUF 插件上的 ReplicatedLinear/专家加载完整性、调试入口和负专家补齐保护。
- `patches/`：可审阅差异；安装使用经过哈希校验的完整增量源文件。
- `provenance/source-map.json`：列出文件来源与整理前源码指纹；`overlay-files.json` 和 `plugin-files.json` 给出安装前后指纹。

上游已有 MTP、CPU offload、压缩权重、候选筛选/复核、CUDA Graph 等能力。本项目的贡献是特定 Q6 / SM80 / RAM PLE 组合的实现、兼容修复和验证，不作“世界首创”或完整算法原创声明。

## 模型、元数据与工具链

模型由 [OrcaRouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) 提供，基础模型和元数据来自 [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。本仓库只提供获取方法、版本与校验值，不捆绑权重、tokenizer、聊天模板或模型配置文件。

基础模型页面标注 Qwen Community License，衍生仓库有自己的模型卡标注和访问要求。部署者应核对所下载版本的实际模型条款及授权；不能用本仓库的 Apache-2.0 代码许可替代模型许可。CUDA、cuFile、驱动、固件和容器基础依赖也分别适用其条款，本项目不打包主机驱动或修改主机许可设置。
