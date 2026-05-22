# -*- coding: utf-8 -*-
"""
Danbooru Tag 数据构建工具

从 xlsx 对照表提取 tag 和中文翻译，输出为 JSON 文件供检索服务使用。

用法：
    python -m core.utils.tag_data_builder
"""
import json
import os
import sys


def build_tag_data(
    xlsx_path: str = None,
    output_path: str = None,
) -> int:
    """
    从 xlsx 文件提取 tag 和 cn 列，写入 JSON。

    Returns:
        写入的 tag 数量
    """
    try:
        import openpyxl
    except ImportError:
        print("需要安装 openpyxl: pip install openpyxl")
        sys.exit(1)

    plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if xlsx_path is None:
        xlsx_path = os.path.join(
            plugin_root,
            "中文化danbooru-tag对照表-词性对AI用优化版-Editor阿巧.xlsx",
        )

    if output_path is None:
        output_path = os.path.join(plugin_root, "data", "danbooru_tags.json")

    if not os.path.exists(xlsx_path):
        print(f"找不到 xlsx 文件: {xlsx_path}")
        sys.exit(1)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active

    tags = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        # 列顺序: id, url, tag, right_tag_cn, ...
        if len(row) < 4:
            continue
        tag = row[2]
        cn = row[3]
        if tag and cn and str(tag).strip() and str(cn).strip():
            tags.append({"tag": str(tag).strip(), "cn": str(cn).strip()})

    wb.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)

    print(f"已导出 {len(tags)} 条 tag 到 {output_path}")
    return len(tags)


if __name__ == "__main__":
    build_tag_data()
