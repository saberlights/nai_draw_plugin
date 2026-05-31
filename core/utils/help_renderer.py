"""`/nai help` 帮助卡片的结构化数据与 HTML 渲染。

把命令分类拆成纯数据，`build_help_html()` 拼出可被 SDK
``render.html2png`` 渲染成 PNG 的页面；HTML 渲染失败时，
``HELP_FALLBACK_TEXT`` 作为纯文本兜底原文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import List, Tuple


@dataclass(frozen=True)
class HelpSection:
    """一组同主题的命令项。"""

    title: str
    items: Tuple[Tuple[str, str], ...]
    hint: str = ""  # 分类下方的额外说明（可多行，用 \n 分隔）
    accent: str = "#ffd56b"  # 分类标题颜色 / 卡片描边渐变主色
    icon: str = "✦"  # 标题左侧装饰 emoji / 字符
    accent2: str = ""  # 渐变副色；留空时由 CSS 自动派生（同色加深）


@dataclass(frozen=True)
class HelpDoc:
    """完整帮助文档结构。"""

    title: str
    subtitle: str
    sections: Tuple[HelpSection, ...]
    footer: Tuple[str, ...] = field(default_factory=tuple)


HELP_DOC: HelpDoc = HelpDoc(
    title="NovelAI 画图插件",
    subtitle="命令速查 · /nai help",
    sections=(
        HelpSection(
            title="生图",
            icon="★",
            accent="#ffd56b",
            accent2="#ff8e72",
            items=(
                ("/nai <描述>", "自然语言生成（中文即可）"),
                ("/nai 随机", "随机生成一张 NSFW 图片"),
                ("/nai 随机自拍", "随机生成一张 NSFW 自拍"),
                ("/nai0 <英文标签>", "直接用英文 tag 生成（不过 LLM）"),
            ),
            hint=(
                "描述含「自拍/镜子/合照/手机拍」走自拍路径；"
                "含「肖像/生活照/portrait」走肖像路径；其它走普通画图。"
            ),
        ),
        HelpSection(
            title="图生图 i2i（§20.1）",
            icon="◆",
            accent="#7ee0ff",
            accent2="#4aa9ff",
            items=(
                ("/nai i2i <描述>", "以参考图为底重绘"),
            ),
            hint=(
                "需先引用一张图或同消息发图；宽高须 64 整除，出图沿用参考图尺寸。\n"
                "可调参数（config.toml）：[i2i] strength / noise"
            ),
        ),
        HelpSection(
            title="Vibe Transfer（§20.3）",
            icon="●",
            accent="#c8a8ff",
            accent2="#7a5cff",
            items=(
                ("/nai vibe存 <名字>", "引用一张图存入 vibe 图库（仅管理员）"),
                ("/nai vibe图库", "列出 vibe 命名图（★ 为选中，仅管理员）"),
                ("/nai vibe删 <名字>", "删除一张 vibe 图（仅管理员）"),
                ("/nai vibe清空", "清空 vibe 图库并重置选定（仅管理员）"),
                ("/nai vibe选 <名字...>", "把默认 vibe 设为 1~4 张（仅管理员）"),
                ("/nai vibe [@<名字...>] <描述>", "用默认 / 单次指定 vibe 出图"),
                ("/nai0 vibe [@<名字...>] <英文>", "同上但直发英文，不过 LLM"),
            ),
            hint=(
                "最多 4 张；全量 cache_id 命中可省 1 anlas。\n"
                "可调参数：[vibe] info_extracted / reference_strength / overall_strength"
            ),
        ),
        HelpSection(
            title="角色参考 Ref（§20.4）",
            icon="◎",
            accent="#ff9ed5",
            accent2="#ff5fa8",
            items=(
                ("/nai ref存 <名字>", "存入 ref 图库（仅管理员）"),
                ("/nai ref图库", "列出 ref 命名图（仅管理员）"),
                ("/nai ref删 <名字>", "删除一张 ref 图（仅管理员）"),
                ("/nai ref清空", "清空 ref 图库（仅管理员）"),
                ("/nai ref选 <名字>", "设定默认 ref 图（仅管理员）"),
                ("/nai ref [@<名字>] <描述>", "用默认 / 单次指定 ref 出图（仅管理员）"),
                ("/nai ref类型 character|style|both", "切换提取目标（仅管理员，会话级）"),
                ("/nai0 ref [@<名字>] <英文>", "同上但直发英文，不过 LLM"),
            ),
            hint=(
                "仅 V4.5 系列模型支持，其它模型自动降级；固定 1 张。\n"
                "可调参数：[character_reference] type / fidelity / strength"
            ),
        ),
        HelpSection(
            title="模型 / 尺寸 / 画师",
            icon="■",
            accent="#7eeab2",
            accent2="#3bd17f",
            items=(
                ("/nai set [代号]", "查看 / 切换模型"),
                ("/nai size [代号]", "查看 / 切换尺寸：竖/v、横/h、方/s"),
                ("/nai art [编号]", "查看 / 切换画师风格预设"),
                ("/nai models", "拉取 NewAPI 网关实时可用模型"),
            ),
            hint=(
                "模型代号：3=V3, f3=Furry V3, 4c=V4 Curated, 4=V4 Full, "
                "4.5c=V4.5 Curated, 4.5=V4.5 Full。会话级，重启回落默认。"
            ),
        ),
        HelpSection(
            title="撤回 / 反推",
            icon="↻",
            accent="#ffb37a",
            accent2="#ff7a3d",
            items=(
                ("/nai on", "开启图片自动撤回（仅管理员）"),
                ("/nai off", "关闭图片自动撤回（仅管理员）"),
                ("/nai 撤回", "撤回最近一张本插件发送的图片"),
                ("/nai 反推", "把引用图反推成 Danbooru tag"),
            ),
            hint=(
                "反推：PNG 原图秒级精确还原；非原图走 WD14 在线 Space 兜底（30~120s 仅正向）。"
            ),
        ),
        HelpSection(
            title="提示词 / NSFW",
            icon="◇",
            accent="#9bc4f5",
            accent2="#5b8def",
            items=(
                ("/nai pt on/off", "开关 prompt 回显"),
                ("/nai nsfw", "查看 NSFW 过滤状态"),
                ("/nai nsfw on/off", "开关 NSFW 过滤（仅管理员，会话级）"),
            ),
        ),
        HelpSection(
            title="管理员",
            icon="▲",
            accent="#ff7ad5",
            accent2="#c93dad",
            items=(
                ("/nai st", "开启管理员模式"),
                ("/nai sp", "关闭管理员模式"),
                ("/nai ban <用户ID>", "拉黑指定用户"),
                ("/nai unban <用户ID>", "取消拉黑"),
                ("/nai banlist", "查看黑名单"),
                ("/nai help", "显示本帮助"),
            ),
        ),
    ),
    footer=(
        "命令均以 /nai 开头；/nai0 直发英文 tag 不过 LLM；",
        "参数中 <...> 必填、[...] 可选、@<名字> 表示引用命名图。",
    ),
)


def _render_section(section: HelpSection) -> str:
    """渲染单个分类卡片。"""

    rows: List[str] = []
    for cmd, desc in section.items:
        rows.append(
            "<li>"
            f'<code class="cmd">{escape(cmd)}</code>'
            f'<span class="desc">{escape(desc)}</span>'
            "</li>"
        )
    hint_html = ""
    if section.hint:
        hint_lines = "<br/>".join(escape(line) for line in section.hint.split("\n"))
        hint_html = f'<p class="hint">{hint_lines}</p>'
    accent2 = section.accent2 or section.accent
    icon_html = (
        f'<span class="icon" aria-hidden="true">{escape(section.icon)}</span>'
        if section.icon
        else ""
    )
    card_style = (
        f"--accent:{escape(section.accent)};"
        f"--accent2:{escape(accent2)};"
    )
    return (
        f'<section class="card" style="{card_style}">'
        f'<h2>{icon_html}<span class="title-text">{escape(section.title)}</span></h2>'
        f'<ul>{"".join(rows)}</ul>'
        f"{hint_html}"
        "</section>"
    )


def build_help_html(doc: HelpDoc = HELP_DOC) -> str:
    """根据 ``HelpDoc`` 生成完整 HTML 字符串。

    样式全部内联，使用本机系统字体；浏览器内不发起网络请求。
    """

    sections_html = "".join(_render_section(s) for s in doc.sections)
    footer_lines = "<br/>".join(escape(line) for line in doc.footer)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>NAI 帮助</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  /* 整体走深色 + 多色高光辉光的"霓虹/赛博"基调 */
  body {{
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "Hiragino Sans GB", "Segoe UI Emoji",
                 "Apple Color Emoji", "Noto Color Emoji",
                 "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", sans-serif;
    background:
      radial-gradient(900px 540px at 12% -8%, rgba(255, 142, 192, 0.32), transparent 60%),
      radial-gradient(820px 620px at 92% 4%, rgba(126, 200, 255, 0.30), transparent 62%),
      radial-gradient(700px 700px at 50% 110%, rgba(167, 139, 250, 0.32), transparent 68%),
      linear-gradient(160deg, #15172a 0%, #1a1d33 50%, #181a2c 100%);
    color: #ecedf5;
    padding: 30px 34px 26px;
    width: 980px;
    position: relative;
    overflow: hidden;
  }}
  /* 网格底纹 + 整页柔光 */
  body::before {{
    content: ""; position: absolute; inset: 0;
    background-image:
      linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
    background-size: 28px 28px, 28px 28px;
    pointer-events: none;
    mask-image: radial-gradient(900px 720px at 50% 30%, #000 30%, transparent 80%);
    -webkit-mask-image: radial-gradient(900px 720px at 50% 30%, #000 30%, transparent 80%);
  }}
  header {{
    margin-bottom: 22px; position: relative; z-index: 1;
    padding: 6px 4px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }}
  header h1 {{
    font-size: 38px; margin: 0 0 8px;
    background: linear-gradient(95deg, #ffd56b 0%, #ff8eb3 35%, #c39bff 65%, #7ee0ff 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-weight: 800; letter-spacing: 0.8px;
    text-shadow: 0 0 24px rgba(195, 155, 255, 0.25);
    filter: drop-shadow(0 2px 14px rgba(255, 142, 192, 0.18));
  }}
  header h1::after {{
    content: " ★"; font-size: 30px; vertical-align: 2px;
    -webkit-text-fill-color: initial;
    color: #ff8eb3;
    text-shadow: 0 0 14px rgba(255, 142, 179, 0.65);
  }}
  header p {{
    margin: 0; font-size: 13.5px; color: #b6bbd2; letter-spacing: 1.2px;
    text-transform: uppercase; font-weight: 500;
  }}
  header p::before {{ content: "◇ "; color: #ffd56b; }}
  header p::after {{ content: " ◇"; color: #7ee0ff; }}
  /* 列式 masonry：浏览器自动平衡两列高度 */
  .grid {{
    column-count: 2; column-gap: 18px;
    position: relative; z-index: 1;
  }}
  /* 卡片：玻璃质感 + accent 渐变描边 + 内发光 */
  .card {{
    position: relative;
    background:
      linear-gradient(180deg, rgba(48, 52, 76, 0.92), rgba(38, 42, 62, 0.92));
    border-radius: 14px;
    padding: 16px 18px 14px;
    margin: 0 0 16px;
    display: block;
    break-inside: avoid;
    -webkit-column-break-inside: avoid;
    page-break-inside: avoid;
    box-shadow:
      0 10px 28px rgba(0, 0, 0, 0.38),
      0 0 0 1px rgba(255, 255, 255, 0.05) inset;
    overflow: hidden;
  }}
  /* 渐变描边：用 border-image + ::before 实现彩色外发光 */
  .card::before {{
    content: ""; position: absolute; inset: 0;
    border-radius: inherit;
    padding: 1.5px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    pointer-events: none;
    opacity: 0.85;
  }}
  /* 卡片顶部高光小弧线 */
  .card::after {{
    content: ""; position: absolute; top: 0; left: 14px; right: 14px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.7;
  }}
  .card h2 {{
    margin: 0 0 10px; font-size: 17px; font-weight: 700;
    color: var(--accent, #ffd56b);
    letter-spacing: 0.5px;
    display: flex; align-items: center; gap: 8px;
    text-shadow: 0 0 14px color-mix(in srgb, var(--accent) 55%, transparent);
  }}
  .card h2 .icon {{
    font-size: 18px; line-height: 1;
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 8px;
    background: linear-gradient(135deg,
      color-mix(in srgb, var(--accent) 25%, transparent),
      color-mix(in srgb, var(--accent2) 25%, transparent));
    border: 1px solid color-mix(in srgb, var(--accent) 50%, transparent);
    box-shadow: 0 0 12px color-mix(in srgb, var(--accent) 35%, transparent);
  }}
  .card h2 .title-text {{
    background: linear-gradient(95deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .card ul {{ list-style: none; padding: 0; margin: 0; }}
  .card li {{
    display: flex; align-items: baseline; gap: 10px;
    padding: 4px 0; line-height: 1.5;
    border-bottom: 1px dashed rgba(255, 255, 255, 0.04);
  }}
  .card li:last-child {{ border-bottom: 0; }}
  .cmd {{
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code",
                 "SF Mono", Consolas, monospace;
    font-size: 12.5px;
    color: color-mix(in srgb, var(--accent) 85%, #ffffff);
    background: linear-gradient(135deg,
      color-mix(in srgb, var(--accent) 14%, transparent),
      color-mix(in srgb, var(--accent2) 10%, transparent));
    padding: 2px 9px; border-radius: 6px; white-space: nowrap;
    flex-shrink: 0;
    border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
    box-shadow:
      0 0 8px color-mix(in srgb, var(--accent) 22%, transparent),
      inset 0 0 0 1px rgba(255, 255, 255, 0.03);
  }}
  .desc {{ font-size: 13px; color: #d5d8e6; }}
  .hint {{
    margin: 10px 0 0; padding: 8px 12px;
    font-size: 12px; color: #c8cce0; line-height: 1.6;
    background: linear-gradient(135deg,
      color-mix(in srgb, var(--accent) 10%, transparent),
      color-mix(in srgb, var(--accent2) 6%, transparent));
    border-left: 2px solid var(--accent);
    border-radius: 6px;
  }}
  footer {{
    position: relative; z-index: 1;
    margin-top: 22px; padding: 12px 14px 4px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 12px; color: #aab0c6; line-height: 1.7;
    text-align: center; letter-spacing: 0.4px;
  }}
</style>
</head>
<body>
  <header>
    <h1>{escape(doc.title)}</h1>
    <p>{escape(doc.subtitle)}</p>
  </header>
  <main class="grid">{sections_html}</main>
  <footer>{footer_lines}</footer>
</body>
</html>"""


def build_help_fallback_text(doc: HelpDoc = HELP_DOC) -> str:
    """渲染失败时使用的纯文本兜底。

    使用与 HTML 同一份结构化数据，避免双份维护。
    """

    lines: List[str] = [f"📖 {doc.title} · {doc.subtitle}", ""]
    for section in doc.sections:
        lines.append(f"【{section.title}】")
        for cmd, desc in section.items:
            lines.append(f"  {cmd} - {desc}")
        if section.hint:
            for hint_line in section.hint.split("\n"):
                lines.append(f"  · {hint_line}")
        lines.append("")
    if doc.footer:
        lines.extend(doc.footer)
    return "\n".join(lines).rstrip()


HELP_FALLBACK_TEXT: str = build_help_fallback_text()


def _render_to_png() -> None:
    """开发期入口：把 ``HELP_DOC`` 渲染成 ``assets/help.png``。

    用法（项目根目录下执行）::

        uv run python -m plugins.nai_draw_plugin.core.utils.help_renderer

    依赖系统已安装至少一种 CJK 字体（如 ``fonts-wqy-microhei``、
    ``fonts-noto-cjk``），否则生成的图片中文会变方框。
    生成的 PNG 直接随插件提交进仓库，运行时不再启动 chromium。
    """

    import asyncio
    import base64 as _base64
    from pathlib import Path as _Path

    from src.services.html_render_service import HtmlRenderRequest, get_html_render_service

    async def _run() -> None:
        request = HtmlRenderRequest(
            html=build_help_html(),
            selector="body",
            viewport_width=980,
            viewport_height=720,
            device_scale_factor=2.0,
            full_page=True,
            omit_background=False,
            wait_until="load",
            wait_for_selector="",
            wait_for_timeout_ms=0,
            timeout_ms=30000,
            allow_network=False,
        )
        result = await get_html_render_service().render_html_to_png(request)
        payload = result.to_payload()
        png_bytes = _base64.b64decode(payload["image_base64"])
        target = _Path(__file__).resolve().parent.parent.parent / "assets" / "help.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(png_bytes)
        print(
            f"[help_renderer] 已写入 {target}，"
            f"{len(png_bytes)} bytes, {payload.get('width')}x{payload.get('height')}"
        )

    asyncio.run(_run())


if __name__ == "__main__":
    _render_to_png()
