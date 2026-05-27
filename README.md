# NovelAI 绘图插件（NewAPI 兼容）

通过 NewAPI 兼容 OpenAI 协议（POST `/v1/chat/completions`）调用 NovelAI 绘图的 MaiBot 插件。

- `/nai` 自然语言生图（LLM 转 Danbooru tag）
- `/nai0` 直接英文标签生图（跳过 LLM）
- `/nai 反推` 图片反推 prompt（PNG 元数据 → WD14 在线 Space 兜底）
- Planner Action 主动出图、reply 后置自动跟图
- 自拍模式（24 关键词、5 类型）、Tag 检索增强、NSFW 过滤、自动撤回、管理员模式

仅文生图，不支持图生图。

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
| `/nai 反推` | 回引/同发一张图，反推 Danbooru tag（原图秒回，非原图走 WD14） |
| `/nai set [3/f3/4c/4/4.5c/4.5]` | 查看/切换模型 |
| `/nai size [竖/横/方]` | 查看/切换尺寸（832x1216 / 1216x832 / 1024x1024） |
| `/nai nsfw [on/off]` | 切换 NSFW 过滤（会话级） |
| `/nai pt on/off` | 切换提示词显示 |
| `/nai on/off` | 切换自动撤回 |
| `/nai st/sp` | 开/关管理员模式（仅管理员） |
| `/nai help` | 帮助 |

`set` / `size` / `nsfw` 都是**会话级**且**运行时临时**，重启后回退到配置默认值。

## 配置

`config.toml` 首次启动自动生成；文件按"先要改的 → 通常不动的"分块，用 `# ========== ... ==========` 大段分隔：

```
[plugin] / [model]                                       # 启用 + NewAPI 网关
[prompt_generator] / [prompt_generator.custom_model]     # 提示词生成 LLM
[action_guard] / [auto_draw_on_reply]                    # 出图触发保护 + reply 跟图
[random_scene] / [tag_retriever]                         # 随机场景 / Tag 检索
[retag]                                                  # 图片反推（PNG 元数据 + WD14 兜底）
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
2. **WD14 在线 Space**（仅当上一步未命中）— 三个 HF Space 串行轮询，前一个失败才打扰下一个，整体最坏 ~120s

```toml
[retag]
enabled = true
cache_ttl_seconds = 3600          # 入站图缓存保留秒数；过期后回引旧图就失效
image_cache_per_stream = 20

wd14_enabled = true               # 非原图时是否调 WD14；需 pip install gradio_client
wd14_model = "SmilingWolf/wd-eva02-large-tagger-v3"
wd14_threshold = 0.35             # 通用 tag 置信度阈值（0~1）
wd14_character_threshold = 0.8    # 角色 tag 置信度阈值
wd14_request_timeout = 35.0       # 单个 Space 超时（实测大图需 16~23s）
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
A: 不支持。

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
