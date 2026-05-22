"""NovelAI Web 图片生成插件。"""

__all__ = ["NaiPicPlugin"]


def __getattr__(name: str):
    if name == "NaiPicPlugin":
        from .plugin import NaiPicPlugin

        return NaiPicPlugin
    raise AttributeError(name)
