# NovelAI 绘图插件（NewAPI 兼容）

通过 NewAPI 兼容 OpenAI 协议（POST `/v1/chat/completions`）调用 NovelAI 绘图的 MaiBot 插件。

- `/nai` 自然语言生图（LLM 转 Danbooru tag）
- `/nai 随机 [角色]` / `/nai 随机自拍 [角色]` 完全随机生成 NSFW 色图；可选指定角色
- `/nai0` 直接英文标签生图（跳过 LLM）
- `/nai i2i` 图生图（引用回复一张图，§20.1）
- `/nai vibe` / `/nai ref` 命名图库：把常用参考图存为命名条目，跨重启保留；vibe = §20.3 风格/氛围迁移，ref = §20.4 角色参考（仅 V4.5）
- `/nai 反推` 图片反推 prompt（PNG 元数据 → WD14 在线 Space 兜底）
- `/nai models` 拉取网关实时可用模型清单
- Planner Action 根据聊天情景主动出图
- 自拍模式（24 关键词、5 类型）、Tag 检索增强、NSFW 过滤、自动撤回、管理员模式

Vibe / 角色参考的命名图库在群聊按 `group_id` 共享、私聊按 `user_id` 隔离，并使用文件系统
存储（图片原始字节落盘可直接查看）；本会话默认选定跨重启保留。Vibe Transfer 还自带本地
`cache_id` 复用：同图同 `info_extracted`
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
/nai 随机 初音未来       # 指定角色，题材/场景/姿势仍完全随机
/nai 随机自拍            # Bot 形象随机自拍
/nai0 1girl, hatsune miku, smile
/nai set 4.5     # 切模型
/nai size 横     # 切尺寸
/nai help
```

## 命令速查

| 命令 | 说明 |
|------|------|
| `/nai <描述>` | 自然语言生图（LLM 生成 prompt） |
| `/nai 随机 [角色]` | 完全随机 NSFW 色图；省略角色时主体也随机 |
| `/nai 随机自拍 [角色]` | 完全随机自拍色图；指定角色时不套用 Bot 形象 |
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
| `/nai ref存 <名字>` | 把引用回复的图存入角色参考图库（**仅管理员**） |
| `/nai ref图库` | 列出角色参考图库（**仅管理员**） |
| `/nai ref删 <名字>` | 删除一张角色参考图（**仅管理员**） |
| `/nai ref清空` | 清空角色参考图库并重置选定（**仅管理员**） |
| `/nai ref选 <名字>` | 设定本会话默认角色参考图（**仅管理员**） |
| `/nai ref [@<名字>] <描述>` | 使用默认或单次指定的角色参考图（仅 V4.5，**仅管理员**） |
| `/nai0 ref [@<名字>] <英文 tags>` | 角色参考直发英文 tags（不过 LLM，**仅管理员**） |
| `/nai ref类型 [character\|style\|both]` | 不带参数查看当前类型；带参数切换 §20.4 提取目标（**仅管理员，会话级**） |
| `/nai 反推` | 回引/同发一张图，反推 Danbooru tag（原图秒回，非原图走 WD14） |
| `/nai set [3/f3/4c/4/4.5c/4.5/5c/5]` | 查看/切换模型（画师预设按模型独立，切换不串用） |
| `/nai size [竖/横/方]` | 查看/切换尺寸（832x1216 / 1216x832 / 1024x1024） |
| `/nai art [编号]` | 查看/切换当前模型的画师风格预设 |
| `/nai models` | 拉取 NewAPI 网关实时可用模型清单 |
| `/nai nsfw [on/off]` | 查看 / 切换 NSFW 过滤（会话级，切换仅管理员） |
| `/nai planner [on/off]` | 查看 / 切换当前会话 Planner 生图（仅管理员，默认开启；关闭后 Planner 不再看到 `nai_web_draw`） |
| `/nai pt on/off` | 切换提示词显示 |
| `/nai tag on/off` | 开关 Danbooru 检索结果回显 |
| `/nai on/off` | 切换自动撤回（仅管理员） |
| `/nai 撤回` | 撤回最近一张本插件发送的图片 |
| `/nai st/sp` | 开/关管理员模式（仅管理员） |
| `/nai ban <用户ID>` / `/nai unban <用户ID>` / `/nai banlist` | 管理黑名单（仅管理员） |
| `/nai help` | 帮助 |

`set` / `size` / `nsfw` / `planner` 都是**会话级**且**运行时临时**；`planner` 重启后默认开启，其余项回退到配置默认值。

**i2i** 走「引用回复一张图」链路（或同条消息附图）；插件按「命令携图 → 引用回引 → 流内最近图」优先级解析。

**vibe / ref** 都走「**命名图库**」：先用 `/nai vibe存 <名字>` 引用一张图入库，再用
`/nai vibe选 <名字1> [<名字2>...]` 设定本会话默认图（vibe 最多 4 张、ref 最多 1 张），
之后 `/nai vibe <描述>` 直接用默认图；也可以 `/nai vibe @<名字1> [@<名字2>...] <描述>`
单次指定。群聊按 `group_id` 共享群图库且群间隔离，私聊按 `user_id` 隔离；每个归属的单库
默认上限为 20 张。所有持久化数据统一落在 MaiBot 根目录下的
`data/plugins/saberlights.nai-draw-plugin/`（宿主为插件分配的统一持久化目录，
旧版 `plugins/nai_draw_plugin/data/` 的存量数据首次加载时自动整体迁移过去）。物理图片分别落在
`data/plugins/saberlights.nai-draw-plugin/named_refs/(groups|users)/<sha256>/(vibe|ref)/<名字>.<ext>`，
可以用文件管理器直接看；选定状态落在同目录 `named_refs/selection.json`，跨重启保留。

`vibe` 的 `cache_id` 缓存落在统一持久化目录的 `vibe_cache.db`，同图同 info_extracted 重发会改写
为 cache_id 复用态省图片编码 + 全量命中省 1 anlas 流量附加费；服务端 cache_id 被淘汰
（§20.3.1 400）时自动清掉本地 stale 条目并提示用户重试。

群聊访问黑/白名单会禁止受限群中的普通 `/nai` 业务命令；`/nai planner [on/off]`
是唯一控制面例外，因此插件管理员仍可在黑名单群中关闭或重新开启 Planner 生图。
该命令仍要求发送者明确列在 `admin.admin_users` 中。

## 配置

`config.toml` 首次启动自动生成；文件按"先要改的 → 通常不动的"分块，用 `# ========== ... ==========` 大段分隔：

```
[plugin] / [model]                                       # 启用 + NewAPI 网关
[prompt_generator] / [prompt_generator.custom_model]     # 提示词生成 LLM
[action_guard]                                           # Planner 出图触发保护
[random_scene] / [tag_retriever]                         # 随机场景 / Tag 检索
[retag]                                                  # 图片反推（PNG 元数据 + WD14 兜底）
[i2i] / [vibe] / [character_reference]                   # 图生图各模式可调参数（§20.1/§20.3/§20.4）
[components] / [prompt_show] / [nsfw_filter]
[auto_recall] / [admin] / [custom_prompt]
[model_defaults]                                         # 全模型共享的生图默认值 + Bot 外貌
[model_v5_full] / [model_v5_curated]                     # 每个模型一节：画师预设 + 质量词 + 参数
[model_v45_full] / [model_v45_curated]
[model_v4_full] / [model_v4_curated]
[model_v3_anime] / [model_v3_furry]
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
| `selfie_prompt_add` / `selfie_negative_prompt_add` | Bot 出镜时追加的固定外貌、身材正向词 / 外貌负向词；历史键名不代表必须自拍 |
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

# local（需要先构建统一持久化目录下的 danbooru_tags.json + 配置 embedding 模型）
top_k = 50
min_score = 0.6
```

`local` 模式首次使用：在插件根目录执行 `python -m core.utils.tag_data_builder` 构建 tag 数据 + `model_config.toml` 配 embedding 模型（如 `bge-m3`），首次会全量 embed 5481 个 tag。

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

> `[character_reference].type` 可以在会话里运行时查看或切换：`/nai ref类型` 查看当前值，`/nai ref类型 character|style|both` 切换（仅管理员）；命令切换的值优先于配置默认。

### 分模型配置（2.0.0 起）

生图参数分两层，后者按当前模型自动选用并覆盖前者：

| 配置节 | 作用 |
|---|---|
| `[model_defaults]` | 全模型共享：采样器、种子、图片格式、质量增强、CFG rescale、兜底尺寸，以及 Bot 外貌（`selfie_*`）的唯一定义 |
| `[model_v5_full]` / `[model_v5_curated]` / `[model_v45_full]` / `[model_v45_curated]` / `[model_v4_full]` / `[model_v4_curated]` / `[model_v3_anime]` / `[model_v3_furry]` | 该模型专属：画师预设、质量词、负面词、尺寸、步数、CFG |

覆盖规则：

- 一般字段：模型节的值直接覆盖 `[model_defaults]`
- `negative_prompt_add`：模型节的值**追加**在共享值之后（不是覆盖），所以通用负面词只写一份
- `selfie_prompt_add` / `selfie_negative_prompt_add`：模型节留空表示继承共享定义，填了才覆盖
- `nai_extra_params`：按 key 逐项覆盖

**每个模型一份独立的画师预设库**，`/nai art` 的选择也按模型分开记，切换模型不会串用。

```toml
[model_defaults]
sampler = "k_euler_ancestral"
seed = -1
image_format = "png"
negative_prompt_add = "{{bad hands}}, watermark"   # 全模型共享，用花括号语法
selfie_prompt_add = "..."                          # Bot 外貌唯一定义

[model_v45_full]
nai_size = "竖图"
num_inference_steps = 28
guidance_scale = 6.0
custom_prompt_add = "location, very aesthetic, masterpiece, no text"   # V4.5 Full 官方配方
negative_prompt_add = "-0.8::feet::"               # 追加在共享负面之后

[[model_v45_full.artist_presets]]
name = "channel风"
prompt = "1.4::kazutake hazano::, 1.2::efe::"
```

为什么按模型 id 而不是按大版本分节：官方文档给出的质量词配方在同版本的 Full 与
Curated 之间就不同（V4.5 Curated 独有 `-0.8::feet::, rating:general`，Furry V3 只用
花括号语法），Curated 的数据集也更小、适用的画师串不一样。

各模型的能力与语法差异：

| 模型 | 多角色 | 角色参考 | 氛围转移 | 数值权重 `1.5::t::` | 负数值 `-1::t::` | SMEA |
|---|---|---|---|---|---|---|
| V5 Full / Curated | ✅ 上限 22 | ❌ | ❌ | ✅ | ✅ | ❌ |
| V4.5 Full / Curated | ✅ 上限 6 | ✅ | ✅ | ✅ | ✅ | ❌ |
| V4 Full / Curated | ✅ 上限 6 | ❌ | ✅ | ✅ | ❌ | ❌ |
| V3 Anime / Furry | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |

> V5 的角色参考与氛围转移是官方尚未随发布上线的能力，不是插件缺失。
> V5 的官方质量词配方 NovelAI 也还没公布，`custom_prompt_add` 默认留空。

从 1.x 升级时插件会在启动时自动迁移 `[model_nai3]` / `[model_nai4]` / `[model_nai4_5]`
到新结构，原文件备份为 `config.toml.pre-2.0.0`。画师预设与各段提示词原样搬运；
若三节的共享标量取值不一致，会按当前 `default_model` 取值并在日志里报出被舍弃的值。

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

**Q: 随机色图怎么指定角色？**
A: 用 `/nai 随机 [角色]`，例如 `/nai 随机 初音未来`；角色会被固定为主体锚点，其余题材、场景、姿势和构图由开放式 LLM 随机生成，并主动避开近期重复。需要自拍构图时用 `/nai 随机自拍 [角色]`；省略角色时沿用 Bot 随机自拍，指定角色后不会注入 Bot 形象。

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
│   ├── rules/             # prompt_rules / selfie_rules
│   ├── services/          # session_state / prompt_memory / tag_retriever 等
│   └── utils/             # 输出解析、后处理、tag_data_builder 等
└── tests/
```

## 许可证

GPL-v3.0-or-later

## 作者

Rabbit

## 更新日志

### v2.0.0 (2026-08-25)

**模型配置按模型 id 分节（配置结构变更，启动时自动迁移）**

- **每个模型一份独立的画师预设库**：`[model_nai3]` / `[model_nai4]` / `[model_nai4_5]` 三节拆成 8 个按模型 id 划分的节（V5 / V4.5 / V4 各含 Full 与 Curated，V3 含 Anime 与 Furry）。此前 V5 直接借用 `[model_nai4_5]`、同版本的 Full 与 Curated 也共用一节，给某个模型调画师串会污染另一个
- **新增 `[model_defaults]` 共享层**：三节取值完全相同的 6 个标量（采样器、种子、图片格式、质量增强、CFG rescale、兜底尺寸）收敛到一处；Bot 外貌（`selfie_*`）也收为唯一定义，修掉"换个模型 Bot 就换发色"（V4.5 是棕发、V4/V3 是黑发）
- `negative_prompt_add` 改为**追加**语义：共享层写通用负面词，模型节只写该版本特有的部分，不必每节重复一遍
- **`/nai art` 的选择按模型隔离**：此前画师索引不带模型维度，V4.5 下选了第 2 个预设再切到 V3，会静默套用 V3 的第 2 个预设，且该模型自己配的 `default_artist_preset` 被完全忽略
- **模型注册表成为能力、约束与官方默认值的单一事实源**（`core/model_capabilities.py`）：加模型只需加一条记录，`/nai set` 代号、可用模型清单、帮助文案、WebUI 下拉、尺寸与步数上限全部随之派生。顺带补上此前缺失的 V5 Curated
- 各模型的 `custom_prompt_add` 默认值改为官方推荐的质量词配方（Full 与 Curated 不同，Furry V3 只用花括号语法）；V5 官方尚未公布配方，默认留空而非照抄 V4.5

**WebUI**

- 页签重排为 连接 / 生图参数 / 画师预设 / 提示词 / 自动化与权限 / 检索与反推；画师预设独占一个页签
- 当前 `default_model` 对应的模型节与预设库自动排到最前并展开、标注"当前默认模型"，其余折叠；此前 group header 里硬编码"你当前默认模型是 V4.5"，改了默认模型就成谎话
- **消灭 38 条填充式字段说明**：`[model_nai4]` / `[model_nai3]` 两节所有字段的说明原本都是"作用同 `model_nai4_5.xxx`"，而 WebUI 的标签取自说明首行——整个"其他模型"页签 36 个字段没有一个可读标签
- `nai_extra_params` 从 TOML 子表改为 `[{key, value}]` 数组，首次可在 WebUI 里编辑（Dashboard 没有 dict 编辑器，子表形态只能藏起来）
- `default_model` 改为下拉（候选来自注册表），`noise_schedule` 改为下拉，`plugin.name` 改为只读

**修复**

- `/nai size` 无参时读的是 `model.default_size`，而 `[model]` 节从来没有这个字段，于是永远显示硬编码的 `832x1216`，与实际生效尺寸脱节
- i2i 参考图尺寸被 `nai_size` 吞掉：`nai_size` 几乎永远非空且排在调用方传入的 `size` 之前，导致 §20.1 的"i2i.image 宽高必须与外层 size 一致"校验必然失败
- V4 及以上不再发送 `autoSmea`：SMEA 是补偿 V1 时代低训练分辨率的采样器变体，V4 起官方不再提供，此前是无效字段
- `quality_toggle = false` 现在会显式发送，此前只在 `true` 时写入字段，用户无法关闭
- 步数兜底从字面量 23 改为该模型注册的推荐值（与 schema 默认 28 不一致已久）
- 角色段数量按模型上限校验（V4/V4.5 为 6，V5 为 22）；`_validate_inner_payload` 用上了此前从未被引用的 `model_name` 形参
- 能力门控的提示语不再硬编码"请先 `/nai set 4.5`"，改为从注册表挑出真正支持该能力的模型

**迁移**

启动时自动把 1.x 的三个模型节搬到新结构，原文件备份为 `config.toml.pre-2.0.0`。画师预设与各段提示词逐字段原样搬运；共享标量取值不一致时按当前 `default_model` 取值，并在日志里报出被舍弃的值。触发条件只看是否存在 1.x 模型节，不看版本号，避免把结构已正确、带用户手写注释的配置重写掉。

### v1.12.0 (2026-07-31)
- **持久化数据迁往宿主统一目录**：全部落盘数据（named_refs 图库、vibe_cache.db、visual_continuity.json、nsfw_state.json、user_blacklist.json、danbooru_tags.json 与 embedding 缓存）从插件源码目录 `plugins/nai_draw_plugin/data/` 迁移到宿主分配的 `data/plugins/saberlights.nai-draw-plugin/`；首次加载时自动整体搬迁存量数据并移除旧目录，宿主的"旧式数据目录"启动警告随之消失。若统一目录已有同名条目则不覆盖、保留原位并在日志指名，留待人工处置
- 新增 `core/data_paths.py` 作为所有 store 默认路径的唯一来源，`on_load` 时与运行时授予的 `ctx.paths.data_dir` 对账，目录布局漂移会在日志里报错暴露
- `tag_data_builder` 改为 `python -m core.utils.tag_data_builder` 方式运行（输出路径跟随统一持久化目录）

### v1.11.0 (2026-07-30)
- **Bot 情景图视觉连续性（服装/场景固定与切换）整体落地**：Tag LLM 输出 v4 分区 JSON（dynamic 动态区 + stable 稳定区），服装与环境稳定 Tag 由程序逐字复用与合成，杜绝 LLM 每轮重译导致的服装/场景漂移；replace 建立的每套服装/环境进入卡片库（各 12 张，LRU 保护当前在用的卡），可随时 switch 逐字切回
- **连续性状态跨重启持久化**：新增 `data/visual_continuity.json`（`VisualContinuityStore`），当前装扮与卡片库落盘；新增 `prompt_generator.visual_state_ttl`（默认 0 = 不过期）单独控制当前服装/环境的沿用期，过期只失效当前装扮、卡片库永续可切回，与 `inherit_ttl`（legacy 继承/自拍上下文）解耦
- **Planner 稳定区协议改为"枚举 + 描述"双字段**：`outfit_change` / `environment_change` 只填 `unchanged|clear|switch|replace`，新内容写 `outfit_new_look` / `environment_new_look`；解析器整句匹配"保持不变/没换/无变化"等口语归 keep，无法归类的自由文本降级为 auto 交 Tag LLM 判定——修复旧版"非枚举文本一律当 replace"把没换装误判成换装的服装漂移主因
- **switch 的 key 解析权下放给 Tag LLM**：Planner 只给口语目标（如 `switch` + "之前那套白裙"）即可，由看得到卡片库的 Tag LLM 选定 key，程序校验必须命中库；库中无匹配（重启丢库/过期）时自动降级 replace 重建，不再整单失败。画图 action 回执现在携带"视觉连续性状态"（当前服装/环境 key + 可 switch 的库清单），Planner 下轮可直接引用真实 key
- **校验与修复循环统一为单一事实源**：删除与 resolve 双标准维护的 `visual_detail_repair_needed` 族，resolve 失败返回结构化原因码（不是 JSON / 缺 rating / switch 未命中 / 首图未建立服装区等），带中文修复指引重试最多 2 次；JSON 截断、缺 rating 等旧版"零重试直接放弃"的失败模式现在都能自愈
- **Bot 出镜强制建立服装稳定区**：首图 keep+空不再被放行（显式 clear 除外），杜绝"服装写进 dynamic 不入缓存、第二张必然重掷"的漏洞；replace 撞已有 key 时按 switch 处理并复用卡片原 Tag，维持"同一 key 永远同一串 Tag"不变量
- **状态提交时序统一**：`last_nai_context`（legacy 继承）与视觉连续性状态一律在图片发送成功后提交；连续性轮次不再写 legacy 继承槽，指定角色（`画指定角色`）与 Bot 情景两条继承链互不污染
- **控制信道与数据信道分离**：`unchanged` / `switch:key` 等控制 token 不再拼进 request 文本，Danbooru 检索 query 与图片 caption 不受枚举噪声污染
- 修复 `BackgroundTaskSupervisor.shutdown()` 在任务同步完成时的事件循环空转死锁（gather 对已终态任务不让出循环，done_callback 的 discard 永远排不上队）
- dynamic 与稳定区去重改为归一化比对（下划线/权重壳同义写法不再漏网）；`inherit_ttl` / `temperature` 配置描述同步实际语义（连续性路径固定 0.2）

### v1.10.0 (2026-05-30)
- **`/nai nsfw` 状态持久化**：把每个 (platform, chat_id) 的开关落盘到 `data/nsfw_state.json`，重启插件后会话级开关自动恢复；优先级 store > 进程内内存 > `nsfw_filter.enabled` 配置默认。`/nai st`/`clear_session_state` 仍会同步清掉持久化条目，语义没变。
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
- Action `nai_web_draw` 的 `description` 单字段拆为 6 个结构化字段：`subject_and_pov` / `action` / `emotion` / `scene_delta` / `dynamic_scene` / `framing`，强制 Planner 分维度思考，避免一锅炖关键词堆砌
- Action 链路下游 LLM 现在能拿到 Planner reasoning（新增 `<<REASONING_CONTEXT>>` 占位符），description 失真时可从 reasoning 救回画面意图，避免动词被软化、情绪被默认套模板
- 新增 `variety_boost` 配置项（V3/V4/V4.5 三段都有），开启后透传到 inner.variety_boost，构图/姿势更随机
- Action Guard 现在主路径同步预检，Planner 第一时间拿到拦截原因；评估结果缓存供后台 handle_action 复用，避免重复读消息库
- 节流间隔大幅调低（explicit 5s / proactive 10s），改为只防同秒重复触发，不再做长冷却

### v1.5.0 (2026-03-23)
- Danbooru Tag 检索增强：embedding 模型从 5481 条中文 tag 对照表语义检索候选标签
- `/nai 随机 [角色]` / `/nai 随机自拍 [角色]`：开放式 LLM 随机生成 NSFW 场景；指定角色时保留角色锚点
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
