# NovelAI 绘图插件（NewAPI 兼容）

通过 NewAPI 兼容 OpenAI 协议（`POST /v1/chat/completions`）调用 NovelAI 绘图服务的 MaiBot 插件。

**核心亮点**：
- 🚀 简单易用：使用 `/nai` 命令 + 自然语言描述即可生图，无需学习复杂语法
- 🤖 智能生成：LLM 自动将中文描述转换为优化的英文提示词
- 📸 自拍模式：智能识别 24 种自拍关键词，支持 5 种自拍类型，自动添加 Bot 形象特征
- ⚡ 自动撤回：可配置图片自动撤回，保护隐私
- 🎭 模型切换：支持快速切换 NAI 3/4/4.5 等不同版本模型
- 🖼️ 尺寸切换：支持快速切换竖图/横图/方图
- 🔞 NSFW 过滤：支持开关 NSFW 内容过滤，灵活控制生成内容
- 🔒 权限控制：支持管理员模式，限制生图命令使用权限

## 功能特性

- ✅ 通过 NewAPI 兼容 OpenAI 协议（`POST /v1/chat/completions`）调用 NovelAI 绘图服务
- ✅ **智能提示词生成**：使用 LLM 自动将自然语言描述转换为优化的英文提示词
- ✅ **命令模式**：`/nai` 命令支持直接输入中文描述，无需掌握 NAI 语法
- ✅ **直接标签模式**：`/nai0` 命令直接使用英文标签生图，跳过 LLM 处理
- ✅ **自拍模式**：智能识别 24 种自拍关键词，支持 5 种自拍类型（手机前置、镜子、高角度、低角度、合照）
- ✅ **模型切换**：支持通过命令快速切换 NAI 3/f3/4/4.5 等模型（会话级别）
- ✅ **尺寸切换**：支持通过 `/nai size` 命令快速切换竖图/横图/方图
- ✅ **画师风格预设**：支持多套画师串预设，可自定义命名，通过配置文件设置
- ✅ **NSFW 内容过滤**：支持 `/nai nsfw on/off` 控制是否过滤 NSFW 内容
- ✅ **提示词显示**：支持 `/nai pt on/off` 控制是否显示生成的提示词
- ✅ **管理员权限控制**：支持开启管理员模式，限制生图命令仅管理员可用
- ✅ **分版本配置**：NAI V3/V4/V4.5 各版本独立配置参数和画师串
- ✅ **自定义 LLM 模型**：支持配置自定义 LLM 模型用于提示词生成
- ✅ **结构化输出**：支持 JSON 格式的 LLM 输出，提升多人场景解析准确性
- ✅ **自拍外貌策略**：可配置自拍模式下的外貌标签处理策略（自动移除/保留/禁用）
- ✅ **提示词后处理**：支持轻量标签排序，优化提示词结构
- ✅ 使用 NAI 格式提示词（大括号权重语法）
- ✅ 文生图功能
- ✅ 支持多种采样器（k_euler, k_euler_ancestral 等）
- ✅ 支持自定义尺寸（竖图、方图或具体尺寸）
- ✅ **自动撤回功能**（可配置延迟时间，支持 `/nai on/off` 控制）
- ✅ **上下文管理**：智能判断是否继承上一轮提示词
- ✅ **Tag 检索增强**：使用 embedding 模型从 5000+ 条 Danbooru 中文对照表中检索候选标签，辅助 LLM 选用更准确的标准 tag
- ✅ **随机生图**：`/nai 随机` 随机生成 NSFW 场景，`/nai 随机自拍` 随机生成 NSFW 自拍
- ❌ **不支持图生图**（仅文生图）

## 安装

1. 将插件文件夹复制到 `plugins/` 目录下
2. 安装依赖：`pip install requests`
3. 编辑 `config.toml` 配置文件（见下方配置说明）
4. 重启 MaiBot

## 快速开始

1. 配置 API 地址和密钥（在 `config.toml` 中）：
   ```toml
   [plugin]
   enabled = true

   [model]
   base_url = "https://api.tuercha.com"   # 由服务方提供
   api_key = "sk-xxxxxxxxxxxxxxxx"          # 由服务方提供
   nai_endpoint = "/v1/chat/completions"
   nai_max_tokens = 100000                  # 单次绘图预算（1 Anlas = 10000 tokens）
   ```

2. 使用 `/nai` 命令开始生图：
   ```
   /nai 画一张初音未来
   /nai 画一个蓝发女仆在花园里
   /nai 自拍，微笑
   ```

3. 或使用 `/nai0` 直接输入英文标签（跳过 LLM）：
   ```
   /nai0 1girl, hatsune miku, smile, masterpiece
   ```

4. （可选）切换模型/尺寸：
   ```
   /nai set 4.5    # 切换到 NAI 4.5
   /nai size 横    # 切换到横图
   ```

5. （可选）查看帮助：
   ```
   /nai help       # 查看所有命令帮助
   ```

## 配置

`config.toml` 第一次启动会自动生成；之后插件加载时如果文件里完全没有 `#` 注释，会按 schema 自动回填注释，已经有注释则尊重用户手写内容。

文件结构按"先要改的 → 通常不动的"分块，用 `# ========== ... ==========` 大段分隔符把同类配置拢在一起，整体顺序大致是：

```
[plugin]                                # 启用/版本
[model]                                 # NewAPI 网关与默认生图模型
[prompt_generator] / [prompt_generator.custom_model]
[action_guard] / [auto_draw_on_reply]   # 出图触发保护
[random_scene] / [random_scene.custom_model]
[tagger] / [tagger.custom_model]
[components] / [prompt_show] / [nsfw_filter]
[auto_recall] / [admin] / [tag_retriever]
[custom_prompt]                         # 自定义系统提示词（不常改，放末尾）
[model_nai4_5] / [[model_nai4_5.artist_presets]]   # 当前默认模型详细参数
[model_nai4]  / [[model_nai4.artist_presets]]      # 作用同 model_nai4_5
[model_nai3]  / [[model_nai3.artist_presets]]      # 作用同 model_nai4_5
```

编辑 `config.toml` 文件：

```toml
[plugin]
enabled = true  # 启用插件

[model]
name = "NovelAI NewAPI Gateway"
base_url = "https://api.tuercha.com"     # NewAPI 兼容网关地址（由服务方提供）
api_key = "sk-xxxxxxxxxxxxxxxx"            # NewAPI 鉴权密钥（由服务方提供）
nai_endpoint = "/v1/chat/completions"      # 固定走 OpenAI 兼容 chat/completions
nai_max_tokens = 100000                    # 单次绘图预算上限（1 Anlas = 10000 tokens）
default_model = "nai-diffusion-4-5-full"   # 默认模型名称
```

### 打标配置（/打标）

`/打标` 会读取你"引用回复"的那条图片消息，对图片做 Danbooru/NAI 风格打标，并输出一行可直接复制给 NAI 的 prompt（角色(作品)+tags）。

推荐用 `[tagger.custom_model]` 单独配置一个支持图像输入的多模态模型，完全独立于其它任务的 `model_task`：

```toml
[tagger]
enabled = true
model_task = "vlm"     # 不用 custom_model 时，走全局 model_config.toml 的哪个任务
temperature = 1.0
max_tokens = 30000     # 请求上限，最终仍受模型自身限制

# 单独配置打标模型（推荐）
[tagger.custom_model]
model_list = ["g3f", "c3f", "u3f"]   # 必须是多模态模型，且 model_list 非空时优先使用
max_tokens = 30000
temperature = 1.0
slow_threshold = 120.0
```

### 分版本模型配置

插件支持为 NAI V3、V4、V4.5 分别配置参数。根据当前使用的模型自动加载对应配置：

```toml
# NAI V3 专用配置
[model_nai3]
nai_size = "832x1216"
sampler = "k_euler_ancestral"
num_inference_steps = 25
guidance_scale = 3.5
custom_prompt_add = "{masterpiece}, best quality, illustration"
negative_prompt_add = "..."
selfie_prompt_add = "..."
selfie_negative_prompt_add = "..."
artist_presets = [
  { name = "风格1", prompt = "artist:example1, artist:example2" },
  { name = "风格2", prompt = "artist:example3, artist:example4" }
]

# NAI V4 专用配置
[model_nai4]
nai_size = "竖图"
sampler = "k_euler_ancestral"
num_inference_steps = 28
guidance_scale = 5.0
custom_prompt_add = ",masterpiece, best quality, absurdres"
negative_prompt_add = "..."
selfie_prompt_add = "..."
selfie_negative_prompt_add = "..."
artist_presets = [
  { name = "风格组合1", prompt = "1.2::artist1::, 1.0::artist2::" },
  { name = "风格组合2", prompt = "1.5::artist3::, 1.0::artist4::" }
]

# NAI V4.5 专用配置
[model_nai4_5]
nai_size = "竖图"
sampler = "k_euler_ancestral"
num_inference_steps = 28
guidance_scale = 5.0
custom_prompt_add = ",masterpiece, best quality, absurdres"
negative_prompt_add = "..."
selfie_prompt_add = "..."
selfie_negative_prompt_add = "..."
artist_presets = [
  { name = "channel风", prompt = "1.4::kazutake hazano::, 1.2::efe::, ..." },
  { name = "简笔朴素", prompt = "1.2::artist:shion(mirudakemann)::, ..." }
]
```

### 重要参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `base_url` | NewAPI 兼容网关地址 | `https://api.tuercha.com` |
| `api_key` | NewAPI 鉴权密钥（OpenAI 风格 Bearer Token） | `sk-xxxx...` |
| `nai_endpoint` | NewAPI 兼容生图端点路径 | `/v1/chat/completions` |
| `nai_max_tokens` | 单次绘图允许消耗的 token 预算（1 Anlas = 10000 tokens） | `100000` |
| `default_model` | NovelAI 默认模型名称 | `nai-diffusion-4-5-full` |
| `nai_size` | 图片尺寸 | `竖图`、`方图`、`1024x1024` |
| `sampler` | 采样器 | `k_euler_ancestral` |
| `num_inference_steps` | 推理步数（上限 28） | `23` |
| `guidance_scale` | 指导强度 | `5.0` |
| `quality_toggle` | 质量增强开关（透传到 inner.qualityToggle） | `true` |
| `auto_smea` | SMEA 自动开关（透传到 inner.autoSmea） | `true` |
| `artist_presets` | 画师风格预设列表 | 见上方配置示例 |

> **注意**：`default_model` 参数是**默认模型**，会话中可通过 `/nai set` 命令临时切换。程序重启后会回退到此默认值。

### 自动撤回配置

```toml
[auto_recall]
enabled = false             # 是否默认启用自动撤回
delay_seconds = 30          # 撤回延迟时间（秒）
id_wait_seconds = 15        # 等待正式消息 ID 的最长时间（秒）
manual_max_age_seconds = 3600  # 手动撤回允许命中的最老图片年龄（秒）；超出视为不可撤回
allowed_groups = []         # 允许使用自动撤回功能的会话白名单
# 示例：allowed_groups = ["qq:123456789", "telegram:987654321"]
```

### 自动出图触发保护（action_guard）

防止 Planner 在不合适的时机连续触发 `nai_web_draw` Action。包含**否定意图兜底**（用户说"不要画"仍调用时拦截）和**频率分级保护**（明确点名 vs Bot 主动出图分两档冷却）。

```toml
[action_guard]
enabled = true                                # 是否启用保护
explicit_request_min_interval_seconds = 45    # 用户明确要求画图 / 自拍时的最小间隔（秒）
proactive_min_interval_seconds = 600          # Bot 主动出图最小间隔（秒），显著高于显式档
weak_negative_ttl_seconds = 60                # 弱否定关键词（"用文字就行"）拦截时效
proactive_self_image_boost = true             # 主动出图且不含自拍/肖像关键词时，自动注入"肖像照 近景"
```

### reply 后置自动跟图（auto_draw_on_reply）

Bot 自己写出的 reply 命中视觉自指 / 情感节点时（"我刚换了新发型"、"今天穿了……"），自动跟一张图。

```toml
[auto_draw_on_reply]
enabled = true               # 开启 reply 后置自动跟图
score_threshold = 0.6        # reply 评分 ≥ 阈值才触发；越高越保守，范围 0.0~1.0
min_interval_seconds = 180   # 独立最小间隔（秒），与显式出图共享一次冷却但独立计时
self_image_boost = true      # 跟图描述不含自拍/肖像关键词时，自动注入对应模式标签
```

### 管理员权限配置

```toml
[admin]
admin_users = ["584232670"]  # 管理员用户ID列表
default_admin_mode = false   # 是否默认启用管理员模式
```

**管理员命令**（仅管理员可用）：
- `/nai st` - 开启管理员模式（仅管理员可生图）
- `/nai sp` - 关闭管理员模式（所有人可生图）

**权限说明**：
- 开启管理员模式后，仅 `admin_users` 中的用户可使用 `/nai` 生图命令
- `default_admin_mode` 设置默认状态，可通过 `/nai st/sp` 动态切换
- 管理员模式是**会话级别**的（群聊/私聊独立配置）

### NSFW 内容过滤配置

```toml
[nsfw_filter]
enabled = false  # 是否默认启用NSFW内容过滤
filter_tags = "{{{{{nsfw}}}}}"  # NSFW过滤标签（高权重），启用时自动添加到负面提示词
```

**说明**：
- 开启后会在负面提示词中添加高权重的 NSFW 标签，禁止生成成人内容
- 同时会在 LLM 提示词生成时注入 NSFW 限制指令，从源头过滤
- 使用 `/nai nsfw on/off` 可在运行时切换（会话级别）

### 提示词显示配置

```toml
[prompt_show]
enabled = false  # 是否默认启用提示词显示（使用 /nai pt on|off 可在运行时切换）
hide_selfie_prompt_add = false  # 自拍模式下是否隐藏配置文件中的自拍补充提示词
```

### 提示词生成配置

插件默认始终使用内置 LLM 生成英文提示词（即使 Planner 提供了 `description` 也会优先改写）。你可以通过 `[prompt_generator]` 区域进行控制：

```toml
[prompt_generator]
model_name = ""          # 指定LLM模型代号，留空则自动选择
temperature = 0.2        # LLM温度
max_tokens = 200         # LLM输出上限
output_format = "text"   # LLM输出格式："text"（默认）或 "json"（结构化输出，适合多人场景）
selfie_appearance_policy = "auto"  # 自拍外貌标签策略："auto"（自动移除LLM随机外貌）/"never"（全部移除）/"keep"（保留所有）
enforce_tag_order = false  # 是否启用轻量标签排序（人数/视角前置，year后置）
# prompt_template = """自定义模板，支持 <<USER_REQUEST>> 和 <<SELFIE_HINT>> 占位符"""

# 自定义模型配置（可选）
# 如果配置了此项，将优先使用自定义模型，而不是系统模型
[prompt_generator.custom_model]
model_list = ["gpt-4o", "claude-3-5-sonnet"]  # 模型列表（按优先级排序）
max_tokens = 20000
temperature = 0.2
slow_threshold = 30.0
```

> `prompt_template` 可选；默认会使用与旧版 `description` 完全一致的生成规则，并且会把用户描述按照"主体→视角→服装→动作→环境→氛围→细节"的顺序重排成结构化文本，再交给 LLM。`<<STRUCTURED_REQUEST>>` 会注入这些槽位内容，`<<USER_REQUEST>>` 则是未经处理的原文，`<<SELFIE_HINT>>` 仅在自拍模式下插入额外指令。

**`output_format` 说明**：
- `text`（默认）：LLM 直接输出逗号分隔的 tag 文本
- `json`：LLM 输出 JSON 结构（`version=2`，包含 `global` 和 `people` 数组），程序自动解析并渲染为多人 `|` 分段格式。适合多人场景，减少 LLM 格式错误

**`selfie_appearance_policy` 说明**：
- `auto`（默认）：自拍模式下，自动移除 LLM 随机生成的外貌标签（发色、发型、瞳色等），保留配置文件中的角色特征。用户明确描述外貌时不移除
- `never`：移除所有外貌标签（包括配置文件中的），仅保留动作、场景、氛围等
- `keep`：保留所有外貌标签，不做处理

### Tag 检索增强配置

插件支持两种检索模式：

- **`online`（默认）**：调用 [DanbooruSearchOnline](https://sakizuki-danboorusearch.hf.space) HF Space，无需本地 embedding 模型，开箱即用；HF Space 冷启动时首次请求可能要 60-90 秒。
- **`local`（回退）**：使用本地 embedding 模型从 Danbooru 中文 tag 对照表（5481 条）检索。需要先生成 `data/danbooru_tags.json` 并配置 `model_config.toml` 的 embedding 模型。

```toml
[tag_retriever]
enabled = true                                            # 是否启用 tag 检索增强
mode = "online"                                           # online = 远程 API，local = 本地 embedding

# --- online 模式（默认） ---
api_url = "https://sakizuki-danboorusearch.hf.space/api"  # 远程 API 地址
timeout = 90.0                                            # 请求超时（秒），HF Space 冷启动约 60-90 秒
search_limit = 30                                         # /search 返回标签上限
search_top_k = 5                                          # /search 每个分词段召回数
related_limit = 20                                        # /related 返回推荐上限
related_seed_count = 8                                    # 从 search 取多少个作为 related 种子
show_nsfw = true                                          # 是否包含 NSFW 标签（会跟随 nsfw_filter 自动调整）
popularity_weight = 0.15                                  # 标签热度对排序的影响权重 (0.0-1.0)

# --- local 模式（回退用） ---
top_k = 50           # 本地检索返回的候选 tag 数量
min_score = 0.6      # 本地检索最低相似度阈值
```

**首次使用 `local` 模式准备**（仅当切到 `mode = "local"` 时需要）：
1. 运行 `python core/utils/tag_data_builder.py` 从 xlsx 对照表生成 `data/danbooru_tags.json`
2. 确保 `model_config.toml` 中已配置 embedding 模型（如 `bge-m3`）
3. 首次启用时会自动调用 embedding API 为所有 tag 构建向量缓存（约 5481 次调用，需几分钟），之后从缓存加载

**工作原理**：
- 用户输入的关键词被拆分后，向检索后端（HF Space 或本地 embedding）请求候选标签
- 检索结果合并去重后，以 `<tag_candidates>` 块注入 LLM 提示模板
- LLM 根据上下文从候选中选择最精确的标准 Danbooru tag，而非自行翻译

## 使用方法

本插件支持多种使用方式：

### 1. 命令模式（推荐）

使用 `/nai` 命令，直接输入自然语言描述，插件会自动使用 LLM 生成符合 NovelAI 格式的提示词：

```
# 基础用法
用户: /nai 画一张初音未来
Bot: [自动生成提示词并生成图片]

# 详细描述
用户: /nai 画一个蓝发女仆在花园里坐着
Bot: [自动生成提示词并生成图片]

# 自拍模式（会自动添加自拍视角和Bot形象特征）
用户: /nai 自拍，微笑
Bot: [生成Bot自拍风格的图片]

# 随机模式（LLM 随机生成 NSFW 场景）
用户: /nai 随机
Bot: [随机生成一张色图]

# 随机自拍（随机 NSFW 场景 + 自拍视角 + Bot形象）
用户: /nai 随机自拍
Bot: [随机生成一张自拍色图]
```

**命令模式特点**：
- 自然语言描述即可，无需掌握 NAI 提示词语法
- 自动使用 LLM 将描述转换为优化的英文提示词
- 支持自拍模式（描述中包含"自拍"、"镜子"、"合照"等 24 种关键词均可触发）
- 支持 5 种自拍类型：手机前置、镜子自拍、高角度俯拍、低角度仰拍、合照
- 自动按照 NovelAI 推荐顺序整理提示词

### 2. 直接标签模式

使用 `/nai0` 命令，直接输入英文标签，跳过 LLM 处理，适合熟悉 NAI 提示词的高级用户：

```
# 直接使用英文标签
用户: /nai0 1girl, hatsune miku, smile, masterpiece, best quality
Bot: [直接使用提示词生成图片]

# 使用 NAI 权重语法
用户: /nai0 {{{masterpiece}}}, {{1girl}}, {{blue hair}}, maid outfit
Bot: [直接使用提示词生成图片]
```

**直接标签模式特点**：
- 跳过 LLM 处理，直接使用输入的标签
- 适合熟悉 NAI 提示词语法的用户
- 可以精确控制提示词内容和权重

### 3. 关键词触发模式

在对话中使用触发关键词，支持自然语言和手动 NAI 格式：

```
# 自然语言（自动转换）
用户: 画一个蓝发女仆
Bot: [自动生成提示词并生成图片]

# 手动 NAI 格式（高级用户）
用户: nai画 {{{masterpiece}}}, {{1girl}}, {{blue hair}}, {{maid outfit}}, sitting
Bot: [直接使用提示词生成图片]
```

### NAI 格式提示词说明

本插件使用 **NovelAI 专用格式**的提示词，使用大括号控制权重：

- `{{{{keyword}}}}` - 极高权重（4层大括号）
- `{{{keyword}}}` - 高权重（3层大括号）
- `{{keyword}}` - 中等权重（2层大括号）
- `keyword` - 常规权重（无括号）
- `[[keyword]]` - 降低权重（中括号）

**示例**：
```
{{{masterpiece}}}, {{blue hair}}, {{maid outfit}}, sitting in garden, sunlight
```

> **提示**：使用命令模式时，无需手动编写 NAI 格式提示词，LLM 会自动处理。

### NSFW 内容过滤功能

支持在群聊或私聊中开关 NSFW 内容过滤：

```
# 开启 NSFW 过滤（禁止生成成人内容）
用户: /nai nsfw on
Bot: ✅ 已在群聊中开启NSFW内容过滤
     🔒 生成的图片将避免包含成人内容
     💡 使用 /nai nsfw off 可关闭过滤

# 关闭 NSFW 过滤
用户: /nai nsfw off
Bot: ✅ 已在群聊中关闭NSFW内容过滤
     🔓 生成的图片将不受NSFW限制
     💡 使用 /nai nsfw on 可重新开启

# 查看当前状态
用户: /nai nsfw
Bot: 当前NSFW过滤状态: 已关闭
```

**NSFW 过滤说明**：
- 开启后会在负面提示词中添加高权重 NSFW 标签
- 同时在 LLM 提示词生成阶段注入限制指令，从源头过滤
- 是**会话级别**的（每个群聊/私聊独立设置）
- 管理员模式开启时，仅管理员可操作

### 自动撤回功能

支持在群聊或私聊中自动撤回生成的图片：

```
# 开启自动撤回
用户: /nai on
Bot: ✅ 已在群聊中开启NAI图片自动撤回功能
     📝 图片将在发送后 5 秒自动撤回
     💡 使用 /nai off 可关闭此功能

# 关闭自动撤回
用户: /nai off
Bot: ✅ 已在群聊中关闭NAI图片自动撤回功能
     💡 使用 /nai on 可重新开启
```

### 尺寸切换功能

支持快速切换图片尺寸（会话级别）：

```
# 查看当前尺寸和可用尺寸列表
用户: /nai size
Bot: 当前使用默认配置尺寸

     可用尺寸:
     竖/v - 竖图 (832x1216)
     横/h - 横图 (1216x832)
     方/s - 方图 (1024x1024)

     使用方法: /nai size <尺寸代号>

# 切换到横图
用户: /nai size 横
Bot: ✅ 已切换到: 横图
     尺寸: 1216x832

# 切换到方图
用户: /nai size s
Bot: ✅ 已切换到: 方图
     尺寸: 1024x1024
```

**注意事项**：
- 尺寸切换是**会话级别**的（每个群聊/私聊独立设置）
- 尺寸设置是**运行时临时的**，程序重启后会回退到配置文件中的默认尺寸
- 所有用户都可以使用 `/nai size` 命令（管理员模式开启时除外）

### 提示词显示功能

支持在生图时显示生成的提示词：

```
# 开启提示词显示
用户: /nai pt on
Bot: ✅ 已开启提示词显示

# 关闭提示词显示
用户: /nai pt off
Bot: ✅ 已关闭提示词显示
```

开启后，每次生图时会先显示 LLM 生成的提示词，方便调试和学习。

### 模型切换功能

支持快速切换 NAI 不同版本的模型（会话级别）：

```
# 查看当前模型和可用模型列表
用户: /nai set
Bot: 当前使用默认模型: nai-diffusion-4-5-full

     可用模型:
     3 - nai-diffusion-3
     f3 - nai-diffusion-3-furry
     4 - nai-diffusion-4-full
     4.5 - nai-diffusion-4-5-full

     使用方法: /nai set <模型代号>

# 切换到 NAI 4.5
用户: /nai set 4.5
Bot: ✅ 已切换到模型: nai-diffusion-4-5-full
     代号: 4.5

# 切换到 NAI 3
用户: /nai set 3
Bot: ✅ 已切换到模型: nai-diffusion-3
     代号: 3

# 切换到 furry 模型
用户: /nai set f3
Bot: ✅ 已切换到模型: nai-diffusion-3-furry
     代号: f3
```

**注意事项**：
- 模型切换是**会话级别**的（每个群聊/私聊独立设置）
- 模型设置是**运行时临时的**，程序重启后会回退到配置文件中的默认模型
- 所有用户都可以使用 `/nai set` 命令（不需要管理员权限）

### 管理员权限控制

支持开启管理员模式，限制只有管理员可以使用生图命令：

```
# 开启管理员模式（仅管理员可执行）
用户: /nai st
Bot: ✅ 已在群聊中开启NAI管理员模式
     🔒 现在仅管理员可使用 /nai 生图命令
     💡 使用 /nai sp 可关闭此模式

# 关闭管理员模式（仅管理员可执行）
用户: /nai sp
Bot: ✅ 已在群聊中关闭NAI管理员模式
     🔓 现在所有人都可使用 /nai 生图命令
     💡 使用 /nai st 可重新开启

# 普通用户在管理员模式下尝试生图
用户: /nai 画一张初音未来
Bot: ❌ 当前会话已开启管理员模式，仅管理员可使用此命令
```

**权限说明**：
- `/nai st` 和 `/nai sp` 命令仅管理员可用
- 管理员模式是**会话级别**的（每个群聊/私聊独立配置）
- 在配置文件中设置 `admin.admin_users` 指定管理员用户ID
- 在配置文件中设置 `admin.default_admin_mode` 可配置默认状态

## 命令速查表

| 命令 | 说明 |
|------|------|
| `/nai <描述>` | 自然语言描述生图（LLM 自动生成提示词） |
| `/nai 随机` | 随机生成一张 NSFW 图片 |
| `/nai 随机自拍` | 随机生成一张 NSFW 自拍图片 |
| `/nai0 <标签>` | 直接使用英文标签生图（跳过 LLM） |
| `/nai set [代号]` | 查看/切换模型（3/f3/4/4.5） |
| `/nai size [代号]` | 查看/切换尺寸（竖/横/方） |
| `/nai nsfw [on/off]` | 查看/切换 NSFW 内容过滤 |
| `/nai pt on/off` | 开关提示词显示 |
| `/nai on/off` | 开关自动撤回 |
| `/nai st/sp` | 开关管理员模式（仅管理员） |
| `/nai help` | 查看帮助信息 |
| `/打标` | 引用回复图片进行打标，输出一行可直接复制给 NAI 的 prompt（角色(作品)+tags） |

## 注意事项

1. **推荐使用命令模式**：使用 `/nai` 命令可以充分利用 LLM 自动生成提示词的功能，更加简单易用
2. **仅支持文生图**：本插件不支持图生图功能
3. **NAI 格式**：如果手动编写提示词，必须使用大括号权重语法，不支持圆括号 `(keyword:1.2)` 格式
4. **API 兼容性**：仅适用于 NewAPI 兼容的 OpenAI 协议网关（POST `/v1/chat/completions`，绘图参数以 JSON 字符串塞入 `messages[0].content`）。当前可用模型固定为 6 个：`nai-diffusion-3` / `nai-diffusion-3-furry` / `nai-diffusion-4-curated` / `nai-diffusion-4-full` / `nai-diffusion-4-5-curated` / `nai-diffusion-4-5-full`。包含 `inpainting` 的模型名不被支持，单次只能生成 1 张
5. **图片格式**：支持返回 URL 或 base64 格式
6. **自拍模式配置**：如需使用自拍模式，建议在配置文件中设置 `selfie_prompt_add` 添加 Bot 的形象特征；如需自拍专属负面词，可设置 `selfie_negative_prompt_add`，它会在触发自拍时追加到 `negative_prompt_add` 后面
## 常见问题

### Q: 推荐使用哪种方式？
A: 推荐使用 `/nai` 命令模式。它会自动使用 LLM 生成优化的提示词，无需掌握 NAI 提示词语法，更加简单易用。

### Q: 如何使用直接标签模式？
A: 使用 `/nai0` 命令，直接输入英文标签，例如：`/nai0 1girl, hatsune miku, smile`。这种模式跳过 LLM 处理，适合熟悉 NAI 提示词的用户。

### Q: 如何切换图片尺寸？
A: 使用 `/nai size <尺寸代号>` 命令：
- `竖` 或 `v` - 竖图 (832x1216)
- `横` 或 `h` - 横图 (1216x832)
- `方` 或 `s` - 方图 (1024x1024)

尺寸切换是会话级别的，重启后会恢复到配置文件中的默认尺寸。

### Q: 如何控制 NSFW 内容？
A: 使用 `/nai nsfw on` 开启 NSFW 过滤（禁止生成成人内容），使用 `/nai nsfw off` 关闭过滤。也可在配置文件 `[nsfw_filter]` 中设置默认状态。

### Q: 如何显示生成的提示词？
A: 使用 `/nai pt on` 开启提示词显示，生图时会先显示 LLM 生成的提示词。使用 `/nai pt off` 关闭。

### Q: 如何查看所有命令帮助？
A: 使用 `/nai help` 命令查看完整的命令帮助信息。

### Q: 如何使用自拍模式？
A: 在 `/nai` 命令描述中包含自拍相关关键词即可自动触发。支持的关键词包括：
- 基础词：自拍、selfie、镜子、mirror
- 动作词：手机拍、前置相机、合照、合影
- 角度词：俯拍、仰拍、高角度、低角度
- 其他：拍照、照镜子、给自己拍等

支持 5 种自拍类型，LLM 会根据描述自动选择最合适的类型：
1. 手机前置自拍（默认）
2. 镜子自拍
3. 高角度俯拍
4. 低角度仰拍
5. 合照自拍

自拍模式会自动添加配置文件中 `selfie_prompt_add` 设置的 Bot 形象特征；如果配置了 `selfie_negative_prompt_add`，也会在触发自拍时追加到系统负面提示词后面。可通过 `selfie_appearance_policy` 配置外貌标签的处理策略。

### Q: 支持图生图吗？
A: 不支持，本插件仅支持文生图。如需图生图，请使用 `custom_pic_plugin` 插件。

### Q: 提示词格式是什么？
A: 如果使用 `/nai` 命令模式，无需关心格式，LLM 会自动处理。如果手动编写提示词，必须使用 NAI 格式（大括号权重），例如 `{{keyword}}`。不支持标准格式 `(keyword:1.2)`。

### Q: 如何设置图片尺寸？
A: 在配置文件中设置 `nai_size = "竖图"` 或 `"方图"`，也可以使用具体尺寸如 `"1024x1024"`。

### Q: 如何使用自动撤回功能？
A:
1. 在配置文件中设置 `auto_recall.enabled = true` 或使用命令 `/nai on` 开启
2. 配置 `delay_seconds` 设置撤回延迟时间
3. 如需限制使用范围，在 `allowed_groups` 中配置白名单
4. 使用 `/nai off` 可临时关闭当前会话的自动撤回

### Q: 如何自定义提示词生成行为？
A: 在配置文件的 `[prompt_generator]` 区域可以：
- 指定使用的 LLM 模型（`model_name`）
- 调整生成温度（`temperature`）
- 设置最大 token 数（`max_tokens`）
- 自定义提示词生成模板（`prompt_template`）
- 配置自定义 LLM 模型（`[prompt_generator.custom_model]`）

### Q: 如何切换生图模型？
A: 使用 `/nai set <模型代号>` 命令，支持的模型代号：
- `3` - NAI Diffusion 3
- `f3` - NAI Diffusion Furry 3
- `4` - NAI Diffusion 4
- `4.5` - NAI Diffusion 4.5

模型切换是会话级别的，重启后会恢复到配置文件中的默认模型。

### Q: 如何启用管理员模式？
A:
1. 在配置文件中设置 `admin.admin_users` 添加管理员用户ID
2. 管理员使用 `/nai st` 命令开启管理员模式
3. 使用 `/nai sp` 可关闭管理员模式
4. 或在配置文件中设置 `admin.default_admin_mode = true` 默认开启

## 项目结构

```
nai_draw_plugin/
├── plugin.py              # 插件入口：注册 Action / Command / Hook，挂 schema 与 config 渲染
├── sdk_runtime.py         # NaiInvocation：单次命令/Action 的运行上下文（生图、自拍、撤回等）
├── runtime_recall.py      # 运行时图片消息追踪 + 撤回 marker 注入
├── legacy_llm_request.py  # 兼容旧 LLM 请求接口的薄封装
├── config.toml            # 配置文件（首次启动自动生成；插件 on_load 时按需回填注释）
├── _manifest.json         # 插件清单
├── core/
│   ├── clients/
│   │   ├── nai_web_client.py        # NewAPI 兼容网关 HTTP 客户端
│   │   └── danbooru_online_client.py
│   ├── commands/                    # 命令实现的散文件，部分逻辑已并入 plugin.py，但测试仍引用
│   ├── mixins/                      # 自动撤回 / 模型配置等可复用 Mixin
│   ├── rules/                       # 提示词与触发规则
│   │   ├── prompt_rules.py
│   │   ├── selfie_rules.py          # 自拍模式（24 关键词、5 类型）
│   │   └── reply_auto_draw.py       # reply 后置自动跟图打分 / 描述合成
│   ├── services/
│   │   ├── session_state.py         # 会话级状态机
│   │   ├── prompt_memory.py         # 上一轮提示词继承
│   │   ├── tag_retriever.py         # tag 检索（online / local 两种模式）
│   │   ├── danbooru_online_retriever.py
│   │   ├── image_generator.py
│   │   └── user_blacklist.py
│   └── utils/
│       ├── prompt_output_parser.py  # LLM 结构化输出解析
│       ├── prompt_postprocessor.py  # 提示词后处理（排序、外貌移除）
│       ├── tagger_utils.py          # /打标 prompt 拼装
│       ├── random_scene_description.py
│       ├── tag_data_builder.py      # xlsx → JSON tag 数据构建（local 模式用）
│       ├── danbooru_api.py
│       └── display_message_helper.py
└── tests/                           # 单元 / 集成测试
```

## 许可证

GPL-v3.0-or-later

## 作者

Rabbit

## 更新日志

### v1.4.0 (2026-05-22) - 配置注释与排版梳理
- 重写 `config.toml` 自动生成器的注释口径：去掉历史遗留的 `NAI VX low-level qualityToggle/autoSmea` / `NovelAI Web` 等字样
- `[model_nai4]` / `[model_nai3]` 段统一为"作用同 `model_nai4_5.xxx`"，避免每个模型段都把同一句话抄三遍
- 新增 `config_file_header`（顶部"建议按这个顺序改"）+ `# ========== ... ==========` 大段分隔符 + V4.5 / V4 / V3 子分隔符
- 明确章节渲染顺序（plugin → model → prompt_generator → action_guard → auto_draw_on_reply → … → model_nai4_5 → V4 → V3），跟 bak 风格对齐
- 渲染两次幂等；schema 外用户自加的 section 仍原样保留

### v1.3.0 (NewAPI 重构)
- 协议切换：底层请求从 NovelAI Web API（GET `/generate`）改为 NewAPI 兼容 OpenAI 协议（POST `/v1/chat/completions`）
- 绘图参数以 JSON 字符串塞入 `messages[0].content`，遵循 NewAPI 文档 §5 字段
- 响应解析改为 markdown data URI 抓图（`![image_N](data:image/png;base64,...)`），并解析 `<!-- seeds:[...] -->` 注释
- 新增 `nai_max_tokens` 配置项（外层 OpenAI 兼容 max_tokens，1 Anlas = 10000 tokens，默认 100000）
- 可用模型清单收敛为 6 个原生模型名：`nai-diffusion-3` / `nai-diffusion-3-furry` / `nai-diffusion-4-curated` / `nai-diffusion-4-full` / `nai-diffusion-4-5-curated` / `nai-diffusion-4-5-full`
- 内层 `n_samples` 固定为 1，包含 `inpainting` 的模型名前置拒绝
- 业务层（命令、提示词生成、自拍、画师预设、撤回、管理员、Tag 检索）与 `nai_pic_plugin` 对齐

### v1.5.0 (2026-03-23)
- 新增 **Danbooru Tag 检索增强**：使用 embedding 模型从 5481 条中文 tag 对照表中语义检索候选标签
- 新增 `/nai 随机` 命令：LLM 随机生成 NSFW 场景并自动生图
- 新增 `/nai 随机自拍` 命令：随机 NSFW 自拍场景 + Bot 形象特征
- 新增 `tag_retriever.py`：Tag 检索服务，支持并发 embedding + 余弦相似度检索 + 向量缓存
- 新增 `tag_data_builder.py`：xlsx 对照表转 JSON 工具
- 新增 `[tag_retriever]` 配置节：enabled/top_k/min_score
- 优化提示词模板：新增 `<tag_candidates>` 候选标签注入，引导 LLM 优先使用标准 Danbooru tag
- 优化 action description 生成：输出关键词形式，提升检索精度

### v1.4.0 (2025-02-03)
- 新增自拍模式增强：支持 24 种触发关键词、5 种自拍类型（手机前置、镜子、高角度、低角度、合照）
- 新增 JSON 结构化输出格式（`output_format = "json"`），提升多人场景解析准确性
- 新增自拍外貌标签策略配置（`selfie_appearance_policy`）：auto/never/keep 三种模式
- 新增轻量标签排序功能（`enforce_tag_order`）：人数/视角前置、year 后置
- 新增提示词显示隐藏自拍补充选项（`hide_selfie_prompt_add`）
- 新增 `selfie_rules.py`：独立的自拍模式规则模块
- 新增 `prompt_output_parser.py`：LLM 结构化输出解析工具
- 新增 `prompt_postprocessor.py`：提示词后处理工具（排序、外貌标签移除）
- 优化 LLM 提示词模板：移除固定词组库，改为更灵活的标签知识指导
- 优化多人 | 分段格式处理逻辑
- 移除提示词默认 1000 字符截断限制

### v1.3.0 (2025-01-28)
- 新增 `/nai nsfw on/off` NSFW 内容过滤功能
- 新增 `[nsfw_filter]` 配置区域
- 重构架构：拆分为 actions/commands/clients/mixins/rules/services/utils 模块

### v1.2.0 (2025-01-23)
- 新增 `/nai0` 直接标签模式，跳过 LLM 处理
- 新增 `/nai size` 尺寸切换命令（竖/横/方）
- 新增 `/nai art` 画师风格切换命令
- 新增 `/nai pt on/off` 提示词显示控制命令
- 新增 `/nai help` 帮助命令
- 新增分版本模型配置（NAI V3/V4/V4.5 独立配置）
- 新增画师串预设命名功能
- 新增自定义 LLM 模型配置支持
- 优化提示词生成模板

### v1.1.0 (2025-12-04)
- 新增模型切换功能（`/nai set` 命令）
- 新增管理员权限控制（`/nai st/sp` 命令）
- 支持会话级别的模型选择
- 支持会话级别的管理员模式控制
- 修复 SSL 证书验证问题

### v1.0.0 (2025-12-03)
- 初始版本
- 支持 NovelAI Web API（std.loliyc.com 等代理接口）
- NAI 格式提示词支持（大括号权重语法）
- 文生图功能
- `/nai` 命令模式，支持自然语言描述
- LLM 智能提示词生成
- 自拍模式（自动添加 Bot 形象特征和自拍视角）
- 上下文管理（智能继承上一轮提示词）
- 自动撤回功能（支持 `/nai on/off` 控制）
- 支持多种采样器和自定义尺寸
