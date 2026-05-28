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
    accent: str = "#ffd56b"  # 分类标题颜色


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
            items=(
                ("/nai <描述>", "自然语言生成（中文即可）"),
                ("/nai 随机", "随机生成一张 NSFW 图片"),
                ("/nai 随机自拍", "随机生成一张 NSFW 自拍"),
                ("/nai0 <英文标签>", "直接用英文 tag 生成（不过 LLM）"),
            ),
            hint=(
                "描述含「自拍/镜子/合照/手机拍」走自拍路径；"
                "含「肖像/生活照/立绘/portrait」走肖像路径；其它走普通画图。"
            ),
        ),
        HelpSection(
            title="图生图 i2i",
            items=(
                ("/nai i2i <描述>", "以参考图为底重绘（§20.1）"),
            ),
            hint=(
                "需先引用一张图或同消息发图。参考图宽高必须 64 整除，"
                "出图沿用参考图尺寸；引用回复部分平台只给缩略图，建议直接附图。"
            ),
        ),
        HelpSection(
            title="Vibe / 角色参考",
            items=(
                ("/nai vibe存 <名字>", "引用一张图存入 vibe 图库"),
                ("/nai vibe图库", "列出当前 vibe 命名图（★ 为选中）"),
                ("/nai vibe删 <名字>", "删除一张 vibe 图"),
                ("/nai vibe清空", "清空 vibe 图库并重置选定"),
                ("/nai vibe选 <名字...>", "把默认 vibe 设为 1~4 张"),
                ("/nai vibe @<名字...> <描述>", "单次覆盖默认选定（1~4 张）"),
                ("/nai vibe <描述>", "用默认 vibe 出图（§20.3）"),
                ("/nai0 vibe [@<名字...>] <英文 tags>", "同上但直发英文，不过 LLM"),
            ),
            hint=(
                "ref 同结构（仅 1 张）：ref存 / ref图库 / ref删 / ref清空 / ref选 / "
                "ref @<名字> / ref <描述>，也有 /nai0 ref。\n"
                "vibe 走 §20.3，最多 4 张，全量 cache_id 命中可省 1 anlas；"
                "ref 走 §20.4，仅 V4.5 系列支持，其它模型自动降级。"
            ),
        ),
        HelpSection(
            title="模型 / 尺寸 / 画师",
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
            title="撤回",
            items=(
                ("/nai on", "开启图片自动撤回（仅管理员）"),
                ("/nai off", "关闭图片自动撤回（仅管理员）"),
                ("/nai 撤回", "撤回最近一张本插件发送的图片"),
            ),
        ),
        HelpSection(
            title="提示词 / NSFW",
            items=(
                ("/nai pt on/off", "开关 prompt 回显"),
                ("/nai nsfw", "查看 NSFW 过滤状态"),
                ("/nai nsfw on/off", "开关 NSFW 过滤（会话级）"),
            ),
        ),
        HelpSection(
            title="反推",
            items=(
                ("/nai 反推", "把引用图反推成 Danbooru tag"),
            ),
            hint=(
                "PNG 原图（NAI/SD 元数据）秒级精确还原；"
                "非原图走 WD14 在线 Space 兜底（20~60s，只输出正向）。"
            ),
        ),
        HelpSection(
            title="管理员",
            items=(
                ("/nai st", "开启管理员模式"),
                ("/nai sp", "关闭管理员模式"),
                ("/nai ban <用户ID>", "拉黑指定用户"),
                ("/nai unban <用户ID>", "取消拉黑"),
                ("/nai banlist", "查看黑名单"),
                ("/nai help", "显示本帮助"),
            ),
            accent="#f08ec0",
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
    return (
        '<section class="card">'
        f'<h2 style="--accent:{escape(section.accent)}">{escape(section.title)}</h2>'
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
  body {{
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "Hiragino Sans GB",
                 "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", sans-serif;
    background: linear-gradient(160deg, #1a1d27 0%, #232636 60%, #1f2230 100%);
    color: #e6e8ee;
    padding: 28px 32px 24px;
    width: 980px;
  }}
  header {{ margin-bottom: 22px; }}
  header h1 {{
    font-size: 28px; margin: 0 0 6px;
    background: linear-gradient(90deg, #ffd56b 0%, #f08ec0 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-weight: 700; letter-spacing: 0.5px;
  }}
  header p {{ margin: 0; font-size: 14px; color: #8a90a3; letter-spacing: 0.6px; }}
  .grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px 16px;
  }}
  .card {{
    background: rgba(42, 46, 58, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 14px 16px 12px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
  }}
  .card h2 {{
    margin: 0 0 8px; font-size: 16px; font-weight: 600;
    color: var(--accent, #ffd56b);
    letter-spacing: 0.4px;
  }}
  .card ul {{ list-style: none; padding: 0; margin: 0; }}
  .card li {{
    display: flex; align-items: baseline; gap: 10px;
    padding: 3px 0; line-height: 1.45;
  }}
  .cmd {{
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code",
                 "SF Mono", Consolas, monospace;
    font-size: 12.5px; color: #7ee0ff;
    background: rgba(126, 224, 255, 0.08);
    padding: 1px 7px; border-radius: 5px; white-space: nowrap;
    flex-shrink: 0;
  }}
  .desc {{ font-size: 13px; color: #c5c8d1; }}
  .hint {{
    margin: 8px 0 0; padding: 7px 10px;
    font-size: 12px; color: #99a0b2; line-height: 1.55;
    background: rgba(126, 224, 255, 0.05);
    border-left: 2px solid rgba(126, 224, 255, 0.35);
    border-radius: 4px;
  }}
  footer {{
    margin-top: 18px; padding-top: 12px;
    border-top: 1px dashed rgba(255, 255, 255, 0.08);
    font-size: 12px; color: #7a8094; line-height: 1.6;
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
