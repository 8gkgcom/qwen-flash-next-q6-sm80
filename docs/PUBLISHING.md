# 上传 GitHub 前的检查

上传的根目录是本项目目录，不是上级工作区。这个目录从明确选择的源码重建，没有复制旧 `.git` 历史。历史测量数据只保留必要的数值和输出哈希。

## 内容与排除项

包含：适配源代码、固定版本信息、许可、构建/启动/测试脚本、中文文档和英文简介、去标识测量记录。

排除：模型权重、tokenizer/模板原文件、服务器地址、GPU UUID、账号密码、访问 Token、个人配置、私有控制台、systemd 原文件、挖矿/电源脚本、运行日志、编译二进制和缓存。`build/`、`runs/`、`data/`、`dist/`、个人虚拟环境及 `config/local*.json` 已加入 `.gitignore`。

审计工具会检查分发文件中的常见密钥格式、私网地址、机器路径、GPU 标识、私钥、字面密码，以及意外的模型/二进制文件。它是模式检查，不能证明所有可能的敏感信息都已识别。新增文件时继续人工审查，尤其不要将整个工作目录拖入 GitHub。

## 本地验证与打包

从干净源码目录打包；若已生成 `config/local.json` 或在根目录放过个人密钥，先将它们移到项目之外。审计即使遇到被 Git 忽略的个人配置也会拒绝，避免误把部署工作目录当作发布目录。

```bash
PYTHONPATH=src Q6_REQUIRE_CPP=1 python3 -m unittest discover -s tests -v
python3 scripts/audit_release.py
python3 scripts/package_release.py
```

打包生成 `dist/flashnext-q6-sm80-0.1.0.zip` 和 SHA-256，归档只含发布源文件，没有 `.git` 历史；`SHA256SUMS` 列出文件校验值。打包拒绝覆盖现有归档，修订后选择新的 `--output dist/新名称.zip`。重新验证代码后再生成归档，避免把旧验证结果配给新源码。

## GitHub 上传

可以将压缩包解压后的项目内容上传到一个新仓库。使用 Git 时，先在**这个目录内**操作：

```bash
git init
git add .
git diff --cached --stat
git diff --cached --check
git status --short
```

确认暂存内容不含个人数据，再使用自己的 Git 身份提交、关联新仓库并推送。项目整理本身不创建远端仓库，也不自动提交或推送。建议首次版本标记为 `v0.1.0` 实验性发布，保留 README 中尚未完成的干净环境 GPU 验证说明。

## 升级规则

不要只改版本号后复用旧测试结论。固定新依赖，审阅差异，验证覆盖层指纹，运行 CPU、真实行/GPU 算子、单请求、混合请求和所需上下文测试。记录失败结果及恢复方案；只有完成对应范围的测试，才扩大支持声明。
