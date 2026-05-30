# NovelAI 绘图插件（NewAPI 兼容）

通过 NewAPI 兼容 OpenAI 协议（POST `/v1/chat/completions`）调用 NovelAI 绘图的 MaiBot 插件。

- `/nai` 自然语言生图（LLM 转 Danbooru tag）
- `/nai0` 直接英文标签生图（跳过 LLM）
- `/nai i2i` 图生图（引用回复一张图，§20.1）
- `/nai vibe` / `/nai ref` 命名图库：把常用参考图存为命名条目，跨重启保留；vibe = §20.3 风格/氛围迁移，ref = §20.4 角色参考（仅 V4.5）
- `/nai 反推` 图片反推 prompt（PNG 元数据 → WD14 在线 Space 兜底）
- `/nai models` 拉取网关实时可用模型清单
- Planner Action 主动出图、reply 后置自动跟图
- 自拍模式（24 关键词、5 类型）、Tag 检索增强、NSFW 过滤、自动撤回、管理员模式

Vibe / 角色参考的命名图库按 `user_id` 隔离、文件系统存储（图片原始字节落盘可直接查看），
本会话默认选定跨重启保留。Vibe Transfer 还自带本地 `cache_id` 复用：同图同 `info_extracted`
重复请求会自动改写成 `cache_id` 复用态，省图片传输 + 编码计费，全量命中还能省 1 anlas 流量
附加费（文档 §20.3.1 / §20.3.2）。

## 使用前提

1. **MaiBot 主程序**：`host_application` 要求 `>=0.10.0`，SDK 要求 `2.x`（见 `_manifest.json`）。
2. **NewAPI 网关账号**：自行准备一个支持 NovelAI 模型转发的 NewAPI 兼容服务（OpenAI 协议、`POST /v1/chat/completions`），拿到 `base_url` + `api_key`。每次绘图按 token 计费（1 Anlas ≈ 10000 tokens）。
3. **一个智商还可以的 LLM 翻译/规划模型**：`/nai` 自然语言生图依赖 LLM 把中文描述转成 Danbooru tag。**强烈建议接 GPT-4o / Claude 3.5 Sonnet / DeepSeek-V3 这一档以上**——低端模型（如 7B 级开源小模型）容易丢主体、瞎编 tag、不遵守 NAI 权重语法，出图质量会断崖式下跌。模型在 `[prompt_generator]` / `[prompt_generator.custom_model]` 里配置，留空则走 MaiBot 全局 task model。
4. **代理（仅当使用 `/nai 反推`）**：WD14 在线兜底走 HuggingFace Space，**国内必须开梯子**。在 `[retag].wd14_proxy` 显式填代理（如 `http://127.0.0.1:7890`），或继承 `HTTPS_PROXY` 环境变量。不开代理基本必超时。若只想用 PNG 元数据反推（无需联网），把 `wd14_enabled = false` 关掉兜底即可。

## 依赖

MaiBot 主程序通常已自带 `httpx` / `requests` / `aiohttp` / `numpy` / `tomlkit`，无需额外装。**仅 `/nai 反推` 的 WD14 兜底需要单独装 `gradio_client`**（软依赖，缺失时 PNG 元数据反推仍可用，非原图直接报错）：

```bash
uv add gradio_client      # 推荐
# 或
pip install gradio_client
```

`[tag_retriever]` 的 `local` 模式（离线 embedding 检索）还需要在 `model_config.toml` 配一个 embedding 模型（如 `bge-m3`），首次会全量 embed 5481 个 tag；online 模式（默认）开箱即用，无额外依赖。

## 安装

1. 复制插件目录到 `plugins/`
2. 按需安装上文「依赖」中的可选包
3. 启动一次 MaiBot 让其生成 `config.toml`，再编辑填入 `base_url` / `api_key` / LLM 模型
4. 重启 MaiBot

## 快速开始

最小配置：

```toml
[plugin]
enabled = true

[model]
base_url = "https://api.tuercha.com"    # 由服务方提供
api_key = "sk-xxxxxxxx"                  # 由服务方提供
nai_endpoint = "/v1/chat/completions"
nai_max_tokens = 100000                  # 1 Anlas = 10000 tokens
default_model = "nai-diffusion-4-5-full"
```

用法：

```
/nai 画一张初音未来
/nai 自拍，微笑
/nai0 1girl, hatsune miku, smile
/nai set 4.5     # 切模型
/nai size 横     # 切尺寸
/nai help
```

## 命令速查

| 命令 | 说明 |
|------|------|
| `/nai <描述>` | 自然语言生图（LLM 生成 prompt） |
| `/nai 随机` / `/nai 随机自拍` | 随机 NSFW 场景 / 自拍 |
| `/nai0 <标签>` | 直接英文标签生图（跳过 LLM） |
| `/nai i2i <描述>` | 图生图（§20.1）：引用一张图，以它为底重绘；宽高须 64 整除 |
| `/nai vibe存 <名字>` | 把引用回复的图存入 vibe 图库（跨重启保留，**仅管理员**） |
| `/nai vibe图库` | 列出 vibe 图库（★ 标记本会话默认选定，**仅管理员**） |
| `/nai vibe删 <名字>` | 从 vibe 图库删一张（**仅管理员**） |
| `/nai vibe清空` | 一键清空 vibe 图库 + 重置本会话选定（不可逆，**仅管理员**） |
| `/nai vibe选 <名字1> [<名字2>...]` | 把本会话默认 vibe 设为 1~4 张（§20.3 `controlnet.images` 最多 4 张，**仅管理员**） |
| `/nai vibe @<名字1> [@<名字2>...] <描述>` | 单次用指定 vibe 图（1~4 张，不动默认选定） |
| `/nai vibe <描述>` | 用本会话默认 vibe 图生图（§20.3） |
| `/nai0 vibe [@<名字...>] <英文 tags>` | 同 `/nai vibe` 但直接发英文 tags，**不过 LLM** |
| `/nai ref存 / ref图库 / ref删 / ref清空 / ref选 / ref @<名字> / ref <描述>` 与 `/nai0 ref ...` | 角色参考族，结构与 vibe 对称但**仅 1 张**（§20.4 `character_references`，仅 V4.5 系列模型，**整族仅管理员可用**） |
| `/nai ref类型 <character\|style\|both>` | 切换本会话 §20.4 提取目标（仅管理员，会话级）；`both` = `character&style` |
| `/nai 反推` | 回引/同发一张图，反推 Danbooru tag（原图秒回，非原图走 WD14） |
| `/nai set [3/f3/4c/4/4.5c/4.5]` | 查看/切换模型 |
| `/nai size [竖/横/方]` | 查看/切换尺寸（832x1216 / 1216x832 / 1024x1024） |
| `/nai models` | 拉取 NewAPI 网关实时可用模型清单 |
| `/nai nsfw [on/off]` | 切换 NSFW 过滤（会话级，仅管理员） |
| `/nai pt on/off` | 切换提示词显示 |
| `/nai on/off` | 切换自动撤回（仅管理员） |
| `/nai st/sp` | 开/关管理员模式（仅管理员） |
| `/nai help` | 帮助 |

`set` / `size` / `nsfw` 都是**会话级**且**运行时临时**，重启后回退到配置默认值。

**i2i** 走「引用回复一张图」链路（或同条消息附图）；插件按「命令携图 → 引用回引 → 流内最近图」优先级解析。

**vibe / ref** 都走「**命名图库**」：先用 `/nai vibe存 <名字>` 引用一张图入库，再用
`/nai vibe选 <名字1> [<名字2>...]` 设定本会话默认图（vibe 最多 4 张、ref 最多 1 张），
之后 `/nai vibe <描述>` 直接用默认图；也可以 `/nai vibe @<名字1> [@<名字2>...] <描述>`
单次指定。图库按 `user_id` 隔离（跨群可用、群间不互相看），默认每库 20 张上限。物理图片
落在 `data/named_refs/users/<sha256>/(vibe|ref)/<名字>.<ext>`，
可以用文件管理器直接看；选定状态落在 `data/named_refs/selection.json`，跨重启保留。

`vibe` 的 `cache_id` 缓存落在 `data/vibe_cache.db`，同图同 info_extracted 重发会改写
为 cache_id 复用态省图片编码 + 全量命中省 1 anlas 流量附加费；服务端 cache_id 被淘汰
（§20.3.1 400）时自动清掉本地 stale 条目并提示用户重试。

## 配置

`config.toml` 首次启动自动生成；文件按"先要改的 → 通常不动的"分块，用 `# ========== ... ==========` 大段分隔：

```
[plugin] / [model]                                       # 启用 + NewAPI 网关
[prompt_generator] / [prompt_generator.custom_model]     # 提示词生成 LLM
[action_guard] / [auto_draw_on_reply]                    # 出图触发保护 + reply 跟图
[random_scene] / [tag_retriever]                         # 随机场景 / Tag 检索
[retag]                                                  # 图片反推（PNG 元数据 + WD14 兜底）
[i2i] / [vibe] / [character_reference]                   # 图生图各模式可调参数（§20.1/§20.3/§20.4）
[components] / [prompt_show] / [nsfw_filter]
[auto_recall] / [admin] / [custom_prompt]
[model_nai4_5] / [model_nai4] / [model_nai3]             # 各版本独立参数 + artist_presets
```

### 重要参数

| 参数 | 示例 |
|------|------|
| `base_url` / `api_key` | NewAPI 网关地址 + 鉴权 |
| `nai_max_tokens` | 单次绘图预算（1 Anlas = 10000 tokens，默认 100000） |
| `default_model` | `nai-diffusion-3` / `-3-furry` / `-4-curated` / `-4-full` / `-4-5-curated` / `-4-5-full` |
| `nai_size` | `竖图` / `方图` / `1024x1024` |
| `num_inference_steps` / `guidance_scale` | 步数（上限 28）/ 指导强度 |
| `quality_toggle` / `auto_smea` / `variety_boost` | 透传到 inner.qualityToggle / autoSmea / variety_boost |
| `artist_presets` | 画师风格预设（多套，可自定义命名） |
| `custom_prompt_add` / `negative_prompt_add` | 固定追加的正/负向词 |
| `selfie_prompt_add` / `selfie_negative_prompt_add` | 自拍模式追加的 Bot 形象 / 负向词 |
| `nai_extra_params` | 透传任意额外字段（如 `cfg_rescale` / `noise_schedule`） |

### 关键配置块

```toml
# 出图触发保护：拦截"用文字就行"等否定意图 + 分级冷却
[action_guard]
enabled = true
explicit_request_min_interval_seconds = 5     # 用户明确要求 → 短间隔
proactive_min_interval_seconds = 10           # bot 主动出图 → 略长
weak_negative_ttl_seconds = 60
proactive_self_image_boost = true             # 主动出图不含自拍/肖像词时自动补"肖像照 近景"

# reply 后置自动跟图：bot 写出"我刚换了新发型"时自动配图
[auto_draw_on_reply]
enabled = true
score_threshold = 0.6
min_interval_seconds = 15
self_image_boost = true

# 自动撤回
[auto_recall]
enabled = false
delay_seconds = 30
id_wait_seconds = 15
manual_max_age_seconds = 3600
allowed_groups = []      # 例：["qq:123456789", "telegram:987654321"]

# 管理员
[admin]
admin_users = ["584232670"]
default_admin_mode = false

# NSFW 过滤
[nsfw_filter]
enabled = false
filter_tags = "{{{{{nsfw}}}}}"

# 提示词显示
[prompt_show]
enabled = false
hide_selfie_prompt_add = false
```

### 提示词生成

```toml
[prompt_generator]
model_name = ""                       # 留空走全局 task model
temperature = 0.2
max_tokens = 200
output_format = "text"                # text / json（多人场景推荐 json）
selfie_appearance_policy = "auto"     # auto / never / keep
enforce_tag_order = false

[prompt_generator.custom_model]       # 可选：单独指定模型
model_list = ["gpt-4o", "claude-3-5-sonnet"]
temperature = 0.2
```

- `output_format`：`json` 输出 `{global, people[]}` 结构，多人场景解析更稳
- `selfie_appearance_policy`：
  - `auto` — 自拍模式下自动移除 LLM 随机生成的外貌 tag，保留配置里的角色特征（用户明确描述外貌时不移）
  - `never` — 移除所有外貌 tag
  - `keep` — 全保留

### Tag 检索增强

```toml
[tag_retriever]
enabled = true
mode = "online"          # online = HF Space，local = 本地 embedding

# online（默认，开箱即用，HF Space 冷启动约 60-90s）
api_url = "https://sakizuki-danboorusearch.hf.space/api"
timeout = 90.0
search_limit = 30
related_limit = 20
show_nsfw = true
popularity_weight = 0.15

# local（需要先构建 data/danbooru_tags.json + 配置 embedding 模型）
top_k = 50
min_score = 0.6
```

`local` 模式首次使用：`python core/utils/tag_data_builder.py` 构建 tag 数据 + `model_config.toml` 配 embedding 模型（如 `bge-m3`），首次会全量 embed 5481 个 tag。

### 图片反推

`/nai 反推`：回引一张图（或在同一条消息内发图带命令）→ 把图反推成 Danbooru tag。**只输出正向 prompt，不返回负面**。

两级链路：
1. **PNG 元数据**（NAI Comment JSON / SD WebUI `parameters`）— 原图秒级精确还原，画师串都齐
2. **WD14 在线 Space**（仅当上一步未命中）— 三个 HF Space 串行轮询，前一个失败才打扰下一个，整体最坏 ~360s

```toml
[retag]
enabled = true
cache_ttl_seconds = 3600          # 入站图缓存保留秒数；过期后回引旧图就失效
image_cache_per_stream = 20

wd14_enabled = true               # 非原图时是否调 WD14；需 pip install gradio_client
wd14_model = "SmilingWolf/wd-eva02-large-tagger-v3"
wd14_threshold = 0.35             # 通用 tag 置信度阈值（0~1）
wd14_character_threshold = 0.8    # 角色 tag 置信度阈值
wd14_request_timeout = 120.0      # 单个 Space 超时（冷启动后首次跑常需 30~90s）
wd14_max_retries = 1
wd14_retry_delay = 0.5

# 关键：国内务必配代理，留空则继承 HTTPS_PROXY 环境变量
wd14_proxy = "http://127.0.0.1:7890"
```

默认轮询三个 HF Space（不渲染进 config.toml，代码内置：`animetimm/dbv4-full-witha-playground` / `pixai-labs/pixai-tagger-demo` / `DraconicDragon/PixAI-Tagger-v0.9-ONNX`）。想换成自部署的 Space，手动加：

```toml
[[retag.wd14_spaces]]
name = "myorg/my-tagger"
type = "pixai"
api = "/predict_image"
```

### 图生图参数（i2i / vibe / 角色参考）

三段对应 NewAPI 文档 §20.1 / §20.3 / §20.4；默认值就是 API 默认，不动也能跑：

```toml
[i2i]
strength = 0.7        # 0.01~0.99，越小越像原图
noise    = 0.0        # 0.0~0.99，注入噪声量

[vibe]
info_extracted     = 0.7   # 每张 vibe 图的 info_extracted（0.01~1.0）
reference_strength = 0.6   # 每张 vibe 图的单独 strength（0.01~1.0）
overall_strength   = 1.0   # ControlNet 整体强度叠加（0.0~1.0）

[character_reference]
type     = "character&style"  # character / style / character&style
fidelity = 1.0                # 0.0~1.0，保真度（次要强度）
strength = 1.0                # 0.0~1.0，主参考强度
```

> `[character_reference].type` 可以在会话里运行时切换：`/nai ref类型 character|style|both`（仅管理员）；命令切换的值优先于配置默认。

### 分版本模型配置

`[model_nai4_5]` / `[model_nai4]` / `[model_nai3]` 三段独立，结构相同。`[model_nai4]` / `[model_nai3]` 字段描述统一为"作用同 `model_nai4_5.xxx`"。每段都可独立配 `artist_presets`：

```toml
[model_nai4_5]
nai_size = "竖图"
sampler = "k_euler_ancestral"
num_inference_steps = 28
guidance_scale = 5.0
custom_prompt_add = ",masterpiece, best quality, absurdres"
negative_prompt_add = "..."
selfie_prompt_add = "..."
artist_presets = [
  { name = "channel风", prompt = "1.4::kazutake hazano::, 1.2::efe::" },
  { name = "简笔朴素", prompt = "1.2::artist:shion(mirudakemann)::" },
]
```

## NAI 提示词格式

手动写 prompt 时必须用大括号权重语法（不支持 `(keyword:1.2)` 标准格式）：

```
{{{{keyword}}}}   极高权重（4 层）
{{{keyword}}}     高权重
{{keyword}}       中等权重
keyword           常规
[[keyword]]       降低
```

NAI 4/4.5 还支持 `1.3::tag::` 数字权重，详见 `core/rules/prompt_rules.py` 的 `<weight_syntax>` 块。

## 常见问题

**Q: 推荐哪种模式？**
A: `/nai` 自然语言模式。LLM 自动转 Danbooru tag，无需掌握 NAI 语法。熟悉 tag 的高级用户用 `/nai0`。

**Q: 自拍模式怎么触发？**
A: 描述里含"自拍 / selfie / 镜子 / 合照 / 手机拍 / 前置相机 / 俯拍 / 仰拍"等 24 个关键词中任一即可。会自动按描述选 5 种类型之一（手机前置 / 镜子 / 高角度 / 低角度 / 合照），并叠加 `selfie_prompt_add` 配置的 Bot 形象特征。

**Q: 怎么定制 LLM 提示词生成？**
A: `[prompt_generator]` 可改 `model_name` / `temperature` / `max_tokens` / `prompt_template`；`[prompt_generator.custom_model]` 可独立指定模型。

**Q: 模型 / 尺寸切换会持久化吗？**
A: 不会。`/nai set` 和 `/nai size` 是会话级且运行时临时，重启回退到 `config.toml` 默认值。

**Q: 支持图生图吗？**
A: 支持。三种链路：
- `/nai i2i <描述>` — §20.1 普通图生图，**引用回复一张图**（宽高须 64 整除，会自动按参考图尺寸出图）
- `/nai vibe ...` — §20.3 Vibe Transfer（风格/氛围迁移，最多 4 张），先 `/nai vibe存 <名字>` 入库，再 `/nai vibe选 <名字1> [<名字2>...]` 设默认（1~4 张）或 `/nai vibe @<名字1> [@<名字2>...] <描述>` 单次指定
- `/nai ref ...` — §20.4 角色参考（仅 V4.5 模型，自动降级），命令族结构与 vibe 对称（存/图库/删/选/@/裸命令）

**Q: `/nai 反推` 经常超时怎么办？**
A: 国内访问 HF Space 必须配代理 — 在 `[retag]` 里填 `wd14_proxy = "http://127.0.0.1:7890"`（或你的代理端口）。如果只想用 PNG 元数据反推（不依赖网络），把 `wd14_enabled = false` 关掉 WD14 兜底，非原图直接返回失败。

## 项目结构

```
nai_draw_plugin/
├── plugin.py              # 插件入口：Action / Command / Hook 注册，schema + config 渲染
├── sdk_runtime.py         # NaiInvocation：单次调用上下文（生图、自拍、撤回、Action Guard 等）
├── runtime_recall.py      # 运行时图片追踪 + 撤回 marker 注入
├── legacy_llm_request.py  # 旧 LLM 请求接口的薄封装
├── config.toml            # 配置文件（on_load 时按需回填注释）
├── _manifest.json
├── core/
│   ├── clients/           # NewAPI 网关 + Danbooru online 客户端
│   ├── commands/          # 命令实现（部分逻辑已合并进 plugin.py）
│   ├── mixins/            # 自动撤回 / 模型配置等可复用 Mixin
│   ├── rules/             # prompt_rules / selfie_rules / reply_auto_draw
│   ├── services/          # session_state / prompt_memory / tag_retriever 等
│   └── utils/             # 输出解析、后处理、tag_data_builder 等
└── tests/
```

## 许可证

GPL-v3.0-or-later

## 作者

Rabbit

## 更新日志

### v1.10.0 (2026-05-30)
- **图生图三段可调参数**：新增 `[i2i]` / `[vibe]` / `[character_reference]` 三段配置，把 NewAPI §20.1（`strength` / `noise`）、§20.3（`info_extracted` / 每图 `reference_strength` / 整体 `overall_strength`）、§20.4（`type` / `fidelity` / `strength`）完整开放给用户；默认值与 API 默认对齐，不动也能跑。`_run_image_pipeline` 改为从 config 读取并兜底夹到合法区间，原本完全没透传的 `i2i.noise` 与 `controlnet.strength` 也补上。
- **新增 `/nai ref类型 <character|style|both>`**：会话级粘性切换 `character_references[i].type`，命令优先于配置默认；`both` 是 `character&style` 的友好别名（仅管理员）。
- **权限收紧**：`/nai ref` 全族（含 `/nai0 ref`）、`/nai vibe` 的 `存 / 选 / 图库 / 删 / 清空` 全部改为仅管理员；vibe 仅 `draw`（`/nai vibe <描述>` / `/nai0 vibe`）对普通用户开放。鉴权与 `/nai nsfw` 同套 `is_admin_user` 判定。
- **WD14 单 Space 超时上限抬到 120s**：原先 `SAFE_SPACE_TIMEOUT_CAP = 35.0` 会把 config 写的 60s 砍回 35s 导致 PixAI-Tagger ONNX 冷启动直接超时；上限调到 120s，整体 deadline 抬到 360s。`wd14_request_timeout` 默认值同步到 120。
- **`/nai help` 卡片改 column 流式布局**：从 2-col grid 换成 column-fill masonry，自动平衡两列高度去掉短卡片下面的死空白；新增「图生图 i2i §20.1」「Vibe Transfer §20.3」「角色参考 Ref §20.4」三张独立卡片，各自附"可调参数 = ..."提示行。
- 修复 `/nai vibe`（vibe 模式）识别 bot 自拍意图并注入 bot 外貌 + `selfie_prompt_add`：仅当 `description` 命中"自拍/肖像"关键词时触发，`raw_prompt`（`/nai0 vibe`）与 ref / i2i 路径保持不注入以免洗掉参考图。`prompt_show.hide_selfie_prompt_add` 命中时显示提示词时隐藏自拍补充。

### v1.9.0 (2026-05-28)
- **Vibe / 角色参考改走命名图库**：先 `/nai vibe存 <名字>` 把图入库（按 `user_id` 隔离，跨群可用），再 `/nai vibe选 <名字1> [<名字2>...]` 设当前会话默认图；之后 `/nai vibe <描述>` 直接用默认图，或 `/nai vibe @<名字1> [@<名字2>...] <描述>` 单次指定。ref 命令族结构对称（仅 1 张）。
- **Vibe 选定支持 1~4 张多图**（§20.3 `controlnet.images` 上限）：`vibe选` 与 `vibe @` 都可接多个名字空格分隔；store 层做硬上限校验（vibe 4 / ref 1），超量统一报错。selection.json 旧版单字符串形态自动升级为列表，无需手动迁移。
- 新增一键清空：`/nai vibe清空` / `/nai ref清空` 删该用户该 scope 的全部图 + 重置该 (scope, user) 在所有 stream 上的选定。
- 新增不过 LLM 版本：`/nai0 vibe [@<名字...>] <英文 tags>` 与 `/nai0 ref [@<名字...>] <英文 tags>`，对应 `/nai0` 的"直发英文 tags"习惯，仍走命名图库的选定/单次覆盖逻辑。
- 修复 Action 链路把"画一张初音未来"误判 bot 自拍：`_compose_description_from_action_data` 不再丢 `description` 字段（之前 5 结构化字段非空就忽略，导致核心锚点"初音未来"被丢）；`handle_action` 的 `is_selfie` 改用 `detect_bot_self_image_intent(raw_description)`，跟 `/nai` 命令链路对齐，不再被 LLM 用作 framing 的 `portrait photo` / `full body portrait` 误命中。
- 新命令：`/nai vibe存` / `/nai vibe图库` / `/nai vibe删` / `/nai vibe选` / `/nai vibe清空` + ref / nai0 对称 11 条
- 命名图库走文件系统（`data/named_refs/users/<sha256>/(vibe|ref)/<名字>.<ext>`，原始字节，文件管理器可直接打开），选定状态用 `data/named_refs/selection.json` 跨重启保留
- 单库容量默认 20 张 / 用户；命名规则：1~32 字符，汉字 + 英文字母 + 数字 + 下划线（禁路径符与 `@`）
- 修复 `/nai i2i` 在引用回复图为缩略图（dims 解不出 / <256）时静默走默认 size 触发上游 400，改为立即拒绝并指引用户改走"同消息附图"
- 修复 `/nai i2i`、`/nai ref`、`/nai vibe` pattern 不兼容引用回复前缀的回归（之前严格的 `(?:.*，说：\s*)?` 起手让 CQ:reply 等前缀匹不上）

### v1.8.0 (2026-05-28)
- 接入 NewAPI 文档新版字段（§5 透传、`/v1/models`、`usage` 上报、`vibe_cache_ids` 解析）
- `config.toml` 注释统一改为"作用 + 可填什么"两段式（整文件扫描幂等）
- 新增图生图族命令：`/nai i2i`（§20.1）、`/nai ref`（§20.4，仅 V4.5）、`/nai vibe`（§20.3）；引用一张图或同消息发图触发
- 新增 `/nai models`：拉取 NewAPI 网关实时可用模型清单
- Vibe Transfer 本地 `cache_id` 复用：把 (图字节 hash, 模型, info_extracted) → cache_id 落盘到独立 SQLite，同图同 info_extracted 重发可走 cache_id 复用态省图片编码 + 1 anlas 流量附加费（全量命中才省附加费，§20.3.2）
- 服务端 cache_id 失效（§20.3.1 400）自愈：错误形态匹配「cache_id 未命中」时自动清掉本次命中的本地 stale 条目并提示重试，避免反复 400
- 修复 `/nai` 把消息里点名二次元角色误判为 bot 自拍的判定

### v1.7.0 (2026-05-23)
- 新增 `/nai 反推`：回引或同发一张图 → 反推 Danbooru tag。两级链路：PNG 元数据（NAI Comment / SD WebUI parameters）秒级精确还原 → WD14 在线 Space 串行轮询兜底（前一个失败才打扰下一个，对 HF 三家友好）
- 只输出正向 prompt，不带出负面（uc / Negative prompt 解析时丢弃）
- 配置项 `[retag]`：`wd14_proxy` 显式配代理（国内必需），`wd14_request_timeout` 默认 35s 覆盖大图 16~23s 实测耗时
- 入站图片缓存 + 引用回复图片解析：两个 EARLY 阶段 hook 缓存所有入站图，命令触发时按"命令携图 → 引用回引 → 流内最近图"优先级解析
- 修复 WD14 客户端构造同步阻塞 event loop（曾导致所有 OBSERVE hook 5s timeout），改走 executor 调用
- `config_hidden_fields` 机制：默认配置生成 / 注释回填时跳过 `wd14_spaces`（用户改不动也不易写对），代码内置默认 3 个 Space，高级用户仍可手动覆盖

### v1.6.0 (2026-05-23)
- Action `nai_web_draw` 的 `description` 单字段拆为 5 个结构化字段：`subject_and_pov` / `action` / `emotion` / `scene_delta` / `framing`，强制 Planner 分维度思考，避免一锅炖关键词堆砌
- Action 链路下游 LLM 现在能拿到 Planner reasoning（新增 `<<REASONING_CONTEXT>>` 占位符），description 失真时可从 reasoning 救回画面意图，避免动词被软化、情绪被默认套模板
- 新增 `variety_boost` 配置项（V3/V4/V4.5 三段都有），开启后透传到 inner.variety_boost，构图/姿势更随机
- reply 后置自动跟图链路新增 `<<REPLY_CONTEXT>>` 透传 bot 回复原文，让图与文匹配
- Action Guard 现在主路径同步预检，Planner 第一时间拿到拦截原因；评估结果缓存供后台 handle_action 复用，避免重复读消息库
- 节流间隔大幅调低（explicit 5s / proactive 10s / auto_draw 15s），改为只防同秒重复触发，不再做长冷却

### v1.5.0 (2026-03-23)
- Danbooru Tag 检索增强：embedding 模型从 5481 条中文 tag 对照表语义检索候选标签
- `/nai 随机` / `/nai 随机自拍`：LLM 随机生成 NSFW 场景（+ Bot 形象）
- `[tag_retriever]` 配置节、xlsx → JSON tag 数据构建工具

### v1.4.0 (2026-05-22)
- 配置注释与排版梳理：`config.toml` 自动生成器去掉历史遗留字样，`[model_nai4]` / `[model_nai3]` 描述统一"作用同 model_nai4_5.xxx"，按 `# ==========` 大段分隔，渲染幂等

### v1.3.0 (NewAPI 重构)
- 协议从 NovelAI Web API → NewAPI 兼容 OpenAI 协议
- 绘图参数以 JSON 字符串塞入 `messages[0].content`，markdown data URI 抓图
- 新增 `nai_max_tokens` 配置项
- 可用模型清单收敛为 6 个原生模型，`inpainting` 模型前置拒绝

### v1.2.0 (2025-01-23)
- `/nai0` 直接标签模式 / `/nai size` 尺寸切换 / `/nai art` 画师风格 / `/nai pt` 提示词显示 / `/nai help`
- 分版本模型配置（NAI V3/V4/V4.5 独立）+ 画师串预设命名 + 自定义 LLM 模型

### v1.1.0 (2025-12-04)
- `/nai set` 模型切换 / `/nai st/sp` 管理员模式（均会话级）

### v1.0.0 (2025-12-03)
- 初始版本：`/nai` 自然语言 + LLM 智能生成、自拍模式、上下文继承、自动撤回
