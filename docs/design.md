# 设计说明：DSA/SWA 混合注意力改造

## 1. 核心思想

SWA = 静态窗口选择器，与 DSA 共用同一条 MLA 稀疏注意力路径。

原实现中 `topk_indices` 有两个来源：本层 indexer 以及 前层传递。我们来给它引入第三个来源：静态滑动窗口。

- 新增 config 字段
  - `hybrid_pattern: list[str]`（逐层 `"dsa"` / `"swa"`）、
  - `sliding_window: int`（默认 4096）
  - `swa_truncate_cache: bool`（默认 True）。
- 对于 `hybrid_pattern[i] == "swa"` 的层：不创建 indexer（其权重按设计丢弃），用静态窗口生成 top-k 索引，送入与原 DSA 完全相同的 scatter→mask→attention 路径
- DSA 层完全不动，行为与原版逐位一致

### 合法性校验（`GlmMoeDsaHybridConfig.__post_init__`，ValueError）

1. `len(hybrid_pattern) == num_hidden_layers`，取值只允许 `"dsa"` / `"swa"`；
2. SWA 层的下一层不允许是 `"shared"` indexer 层（SWA 不传 top-k，链条会断；
   SWA 层 forward 返回的 topk_indices 由 M4 置 None）；
3. SWA 层本身的 `indexer_types[i]` 无约束（full/shared 均可改造）；
4. `swa_truncate_cache=True` 时 `hybrid_pattern[0]` 必须为 `"dsa"`；
5. 截断模式不支持 chunked prefill（seq_len>1 且 cache 非空），运行时显式 `raise NotImplementedError`）。

## 2. 权重复用分类

| 类别 | 参数 | 说明 |
| --- | --- | --- |
| 直接复用 | `embed_tokens`、`lm_head`、所有 LayerNorm、MLP/MoE 全部、注意力的 `q_a/q_b/kv_a/kv_b/o_proj` |  |
| 主动丢弃 | SWA 层的 `self_attn.indexer.{wq_b, wk, weights_proj, k_norm.*}`| 功能被静态窗口替代；weight_report 标记为 `dropped_by_design` |
| 重新初始化 | 仅 `model.norm.weight` | 只是为了验证功能没必要专门下载一片权重 |

## 3. 代码改动（相对 transformers 5.16.1 `glm_moe_dsa`）

| 改造点 | 位置 | 内容 |
| --- | --- | --- |
| 1 | `configuration.py` | 新增三个字段 + 五条校验；`model_type="glm_moe_dsa_hybrid"`，注册 Auto 类 |
| 2 | `modeling.py` `Attention.__init__` | SWA 层 `indexer=None` |
| 3 | 新增 `_window_topk_indices` | 静态窗口索引 `[B,S,w]` |
| 4 | `Attention.forward` | SWA 分支逻辑 |
| 5 | `DecoderLayer.forward` 末尾 | SWA 层返回 `topk_indices=None` |
| 6 | `Attention.forward` | KV cache 截断，详情见下 |

## 4. KV Cache 截断的四个坐标与形状问题
(a) 时序不对称：prefill 和 decode 截断逻辑不一样，decode 容易差一位。prefill 用完整KV做计算，算完再裁缓存；decode 更新KV后立刻切到最后w个token，保证窗口边界不出错。

(b) mask对齐：全局mask是按最长序列生成的，SWA截断后KV更短，需要把mask尾部裁到和当前KV长度对齐。

(c) 分支优先级：先判断缓存是否为空。开启截断时，长序列prefill直接报错；单token解码先裁缓存；其余场景用绝对坐标生成窗口索引。

(d) 位置簿记：开启截断后，SWA缓存长度不等于总输入token数，因此强制要求第0层必须是DSA。

配套开关：`swa_truncate_cache`
- `False`: 只加mask、不裁缓存，用于等价校验；
- `True` 裁剪缓存省显存
已有测试验证两种模式输出一致。

## 5. 为什么不用 ransformers 原生滑动窗口组件
1. 接入滑动窗口要改mask分发逻辑，混合层KV长度不一致，mask语义复杂难调。
2. 我们优先让SWA复用DSA整套「索引→scatter→mask→注意力」通路，二者逻辑同构，等价性测试更干净、排错简单。

代价就是需要自己处理截断时序、mask对齐、位置簿记，这个代价我还是可以接受的。