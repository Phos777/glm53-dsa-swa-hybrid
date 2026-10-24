# GLM-5.3 DSA/SWA 混合注意力改造

## 题目描述

基于 **GLM-5.3**，调研其模型架构、公开代码及训练框架，**将原始注意力改造为 DSA（Dynamic Sparse Attention）与 SWA（Sliding Window Attention）混合注意力架构，并实现原始预训练权重的分模块加载与复用**。重点考察你的模型架构理解、代码阅读与修改、权重迁移及工程验证能力。

## 核心思路

用 SWA 做静态窗口选择器，与 DSA 共用同一条"索引 → scatter → mask → MLA 注意力"的完整推理路径。SWA 层丢弃 indexer，增加静态窗口索引生成，不引入任何新参数。除 SWA 层 indexer 外，其他的全部权重复用。

本项目基于 transformer 5.16.1 的 glm_moe_dsa 来进行实现。只切两层的权重来演示思路。

## 仓库结构

```
├── docs/
│   ├── design.md                 # 设计思路详细阐述
│   └── experiment_report.md      # 切片 / 权重报告 / 测试 / bench / 训练实验结果与分析
├── src/glm53_hybrid/
│   ├── configuration.py          # GlmMoeDsaHybridConfig
│   ├── modeling.py
│   └── weight_loader.py
├── scripts/
│   ├── slice_checkpoint.py       # FP8→bf16 反量化切片工具
│   ├── convert_to_hybrid.py      # 生成 hybrid config/checkpoint + weight_report
│   ├── train.py                  # 冒烟测试
│   ├── infer.py                  # 冒烟测试
│   └── bench.py
├── tests/test_hybrid.py
└── results/                      # weight_report / bench / train_log / slice_report
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 模型切片，需要 GLM-5.3 的如下文件放在 model/ 路径下
# - config.json
# - generation_config.json
# - model-00001-of-00141.safetensors
# - model.safetensors.index.json
# - tokenizer_config.json
# - tokenizer.json
python scripts/slice_checkpoint.py --src model --dst checkpoints/glm53-mini-2l

# 2. 转换为 hybrid 模型
python3 scripts/convert_to_hybrid.py --pattern dsa,swa --window 4096

# 3. 测试。功能性测试不需要真实权重；带权重的测试需要第 1 步的切片
python -m pytest tests/ -m "not slow"   # 7 个功能测试，CPU 约 6s
python -m pytest tests/ -m slow         # 3 个权重测试，约 1min

# 4. 推理 / 性能 / 训练
PYTHONPATH=src python scripts/infer.py --max-new-tokens 16
PYTHONPATH=src python scripts/bench.py
PYTHONPATH=src python scripts/train.py --pattern dsa,swa --tag after
```

## 结果摘要

我们拿两层的小模型做验证，配置是一层 DSA、一层 SWA：一共 37 个权重键，其中 32 个可以直接复用原模型权重；只有第 1 层原本 DSA 用的索引模块要删掉，这部分大概 9.37M 参数，没有缺权重、也没有多出多余权重。

功能上一共 10 项测试全部跑通：包括普通解码、截断解码的数值对齐；当输入序列长度落在窗口范围内时，SWA 的计算结果和全量注意力完全一致；窗口以外的注意力分数会被强制置零；KV 缓存截断逻辑也符合预期。

显存方面，同样两层小模型、32k 序列长度，两层都用 DSA 时 KV 缓存占 4.02 GiB；改成 DSA+SWA 再加缓存截断之后降到 2.26 GiB，显存省了 44%。78 层完整模型的显存预估，在实验报告里单独算了。

训练侧只做了冒烟验证。冻结嵌入层、输出头和注意力模块，只训练 MLP 部分（0.45B 参数）。设置窗口大小128、块大小256、DSA topk=128，保证两条稀疏注意力分支都能真正跑起来，改造前后 loss 都能正常往下走。

详见 `docs/experiment_report.md`。

## 已知限制

1. 模型的 norm 权重是手动初始化成全 1 的，当前这个小模型只作为模型结构层面的展示，不能代表原版 GLM-5.3 的真实权重。 
2. 短输入下 DSA 会退化成全量注意力，稀疏带来的显存 / 速度收益没法在短序列上体现，收益数据是根据 bench 结果外推的。
3. SWA 层暂时对接不了 flash-mla 加速核，如只是跑 eager 没问题，走其他加速核会有问题。
4. 精度对比说明：能拿到的公开权重只有 FP8 版本，切片的时候做了 FP8 转 bf16 反量化。所有改造前后对比，都是在这份转完的权重上做的，内部对比没问题，但这个精度不能直接对标官方内部的原始基线。
