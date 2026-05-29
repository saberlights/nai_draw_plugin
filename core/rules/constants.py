# -*- coding: utf-8 -*-
"""
规则常量统一管理

集中管理所有硬编码的关键词、tag 列表、禁用规则等常量，
供 prompt_rules / selfie_rules / reply_auto_draw 统一引用。
"""

import re

# ==================== 质量词与通用禁用 ====================

# 质量词禁止列表（由系统自动添加，LLM 不应输出）
QUALITY_TAGS_FORBIDDEN = [
    "masterpiece",
    "best quality",
    "high quality",
    "ultra quality",
    "amazing quality",
]

# 画师 tag 禁止（由系统自动添加）
ARTIST_TAG_PATTERN = re.compile(r"artist:\w+")

# 通用禁止的 tag
FORBIDDEN_TAGS_COMMON = [
    "selfie stick",
    "holding selfie stick",
]


# ==================== SFW/NSFW 禁用 tag ====================

# SFW 模式硬性禁用 tag - 性器/裸露
SFW_FORBIDDEN_EXPLICIT = [
    "nsfw", "nude", "naked", "sex", "penis", "pussy", "vagina",
    "nipples", "anus", "penetration", "cum", "ejaculation",
    "fellatio", "cunnilingus", "paizuri", "footjob", "handjob",
    "masturbation", "orgasm", "topless", "bottomless",
]

# SFW 模式硬性禁用 tag - 擦边/性暗示
SFW_FORBIDDEN_SUGGESTIVE = [
    "cleavage", "suggestive", "seductive", "bikini", "lingerie",
    "swimsuit", "panties", "thong", "underwear", "cameltoe", "see-through",
]

# SFW 模式所有禁用 tag（合并）
SFW_FORBIDDEN_ALL = SFW_FORBIDDEN_EXPLICIT + SFW_FORBIDDEN_SUGGESTIVE


# ==================== 自拍/肖像触发词 ====================

# 用户原话明确要求看图/画图/发图/自拍/肖像/追图
EXPLICIT_IMAGE_REQUEST_KEYWORDS = [
    # 直接画图/出图请求
    "画图", "画一", "画个", "画张", "生成图", "出图", "出一张", "发图", "配图",
    "来一张", "来张", "来一个", "来个", "整张", "整一张", "整一个", "整个",
    "给我画", "给我来", "给我发", "给我看", "给我整", "帮我画", "帮我生成",
    "再来一张", "再来个", "另一张", "再画一张", "再发一张", "重新画", "重新发",
    # 自拍/肖像/照片
    "自拍", "selfie", "自拍照", "镜子", "镜拍", "镜中", "前置", "前摄", "合照", "合影",
    "拍给我看", "拍一张", "拍张", "肖像", "portrait", "立绘", "证件照", "生活照",
    "头像照", "candid",
    # 想看 bot 本人
    "看看你", "看你", "想看你", "发你", "发张你的", "你长什么样", "你的照片",
    "你的样子", "你今天穿了什么", "你今天的样子", "你穿什么", "看看黑丝", "看看白丝",
    "你的腿", "你的脚", "你的鞋", "你的脸", "你的全身",
    # 追图/换装
    "换个角度", "换个姿势", "换个背景", "换个场景", "再拍", "同一套", "这套", "这身",
]

# 用户原话明确不要图/不要画 - 强拒绝
NEGATIVE_IMAGE_INTENT_KEYWORDS_STRONG = [
    "不要画", "不用画", "别画", "别画图", "不画了",
    "不要图", "不用图", "别发图", "不要发图", "别配图", "不用配图",
    "别给我画", "别给我发", "不要给我画", "不要给我发",
    "不用给我画", "不用给我发",
]

# 用户原话明确不要图/不要画 - 弱拒绝
NEGATIVE_IMAGE_INTENT_KEYWORDS_WEAK = [
    "文字就行", "文字回复就行", "文字说就行", "用文字",
]

# 合并视图（兼容旧引用）
NEGATIVE_IMAGE_INTENT_KEYWORDS = (
    NEGATIVE_IMAGE_INTENT_KEYWORDS_STRONG + NEGATIVE_IMAGE_INTENT_KEYWORDS_WEAK
)

# 自拍触发关键词：用户明确要自拍构图
SELFIE_TRIGGER_KEYWORDS = [
    # 直接自拍
    "自拍", "selfie", "self-shot", "自己拍", "给自己拍", "自拍照",
    # 镜子相关
    "镜子", "mirror", "照镜子", "镜中", "镜面", "浴室镜", "全身镜", "穿衣镜",
    # 手机拍照
    "手机拍", "前置", "前置摄像头", "front camera", "举手机",
    # 合照自拍
    "合照", "合影", "一起拍", "group selfie",
]

# 肖像触发关键词：用户想要 bot 本人的照片，但不要自拍
PORTRAIT_TRIGGER_KEYWORDS = [
    "肖像", "肖像照", "肖像画", "portrait",
    "头像", "头像照",
    "生活照", "证件照", "立绘",
    "candid",
]

# 「想看 bot 本人」表达：隐式 self-image 请求
BOT_SELF_REFERENCE_KEYWORDS = [
    "看看你", "看你", "想看你", "发你", "发张你的",
    "你长什么样", "你的照片", "你的样子", "你今天穿了什么", "你今天的样子",
    "你穿什么", "你的脸", "你的全身",
    "看看黑丝", "看看白丝", "你的腿", "你的脚", "你的鞋",
    "拍给我看",
]

# selfie 后处理触发关键词集（合并）
BOT_SELF_IMAGE_INTENT_KEYWORDS = (
    SELFIE_TRIGGER_KEYWORDS
    + PORTRAIT_TRIGGER_KEYWORDS
    + BOT_SELF_REFERENCE_KEYWORDS
)

# LLM 输出中表示自拍的标签
SELFIE_OUTPUT_TAGS = [
    "selfie", "mirror selfie", "group selfie",
]

# LLM 输出中表示肖像的标签
PORTRAIT_OUTPUT_TAGS = [
    "portrait photo", "candid photo",
    "upper body portrait", "full body portrait",
]


# ==================== reply_auto_draw 触发词 ====================

# 强信号：bot 自指视觉细节（穿/坐/躺/动作/位置/即时状态）
SELF_VISUAL_KEYWORDS = (
    # 服装 / 着装动作
    "我穿", "我换上", "穿着", "穿了", "脱下", "脱了", "换上", "披着", "戴着", "戴了",
    # 姿态 / 动作
    "我坐", "我躺", "我靠", "我趴", "我蹲", "我站", "我抱", "我抬", "我伸", "我抓",
    "我笑", "我哭", "我累", "我饿", "我困", "我冷", "我热",
    # 即时状态
    "我刚", "我现在", "我在", "我正", "我刚刚",
    "刚洗完", "刚洗澡", "刚出浴", "刚起床", "刚醒", "刚到家", "刚回来", "刚吃完",
    "刚做完", "刚下班", "刚下课", "刚化完妆", "刚换好",
    # 身体局部自指
    "我的头发", "我的发型", "我的腿", "我的脚", "我的手", "我的脸", "我的眼睛",
)

# 中信号：情感节点 / 亲密互动
EMOTIONAL_BEAT_KEYWORDS = (
    "晚安", "早安", "早上好", "想你了", "好想你", "想见你", "抱抱", "亲亲", "么么哒",
    "回来了", "到家了", "回家了", "我回来啦", "我到啦",
    "吃饱了", "吃完啦", "睡了", "睡觉啦", "困了", "累了一天",
    "在想你", "陪我", "陪你", "等你", "等等我",
)

# 场景 / 地点 / 活动信号
SCENE_KEYWORDS = (
    "窗边", "阳台", "床上", "沙发", "厨房", "浴室", "书桌", "桌前",
    "咖啡店", "便利店", "超市", "餐厅", "公园", "地铁", "公交", "学校", "教室", "图书馆",
    "路上", "回家路上", "下班路上", "便道", "海边", "山上",
    "外面下雨", "下雪了", "天黑了", "天亮了", "夕阳",
)

# 强负向：bot 在做理性 / 工具性回答
DISQUALIFYING_PATTERNS = (
    re.compile(r"```"),                                                       # 代码块
    re.compile(r"^\s*\d+[.)、]\s", flags=re.MULTILINE),                        # 行首 "1. " / "1) " / "1、"
    re.compile(r"^\s*[-•·]\s", flags=re.MULTILINE),                            # 列表项
    re.compile(r"(?:首先|其次|另外|最后|总结一下|综上|换句话说)"),
    re.compile(r"(?:函数|参数|变量|API|接口|算法|数据库|sql|json|http)", re.IGNORECASE),
)

# 弱负向：bot 在询问 / 不确定 / 解释
SOFT_NEGATIVE_KEYWORDS = (
    "你觉得呢", "你怎么看", "我不太懂", "我不太清楚", "我想想",
    "这个我", "这个怎么", "为什么", "怎么办",
)


# ==================== 发色/瞳色冲突检测 ====================

# 发色/发型相关关键词（用于冲突检测）
HAIR_RELATED_KEYWORDS = [
    " hair", "haired", "twintails", "twin tails", "ponytail", "side ponytail",
    "braid", "pigtails", "bun", "bob cut", "hime cut", "bangs", "forelock",
    "ahoge", "side lock", "side locks", "hairclip", "hair clip", "barrette",
    "hair ornament", "hair ribbon", "hair bow", "hairband", "headband",
    "scrunchie", "wavy ends", "loose hair strands", "pixie cut", "cropped hair",
    "short bob", "bob haircut", "shoulder-length hair", "chin-length hair",
]

# 瞳色相关颜色集合
EYE_COLOR_KEYWORDS = {
    "black", "brown", "blue", "red", "green", "purple", "orange",
    "gray", "grey", "golden", "yellow", "pink", "aqua", "cyan",
}

# 瞳色相关特殊 tag
EYE_SPECIAL_TAGS = {"eyelashes", "long eyelashes", "heterochromia"}


# ==================== Framing 三档互斥 ====================

# Framing 三档（只能选一个）
FRAMING_CLOSE_UP = ["close-up", "portrait"]  # 特写：头肩特写
FRAMING_HALF_BODY = ["upper body", "cowboy shot"]  # 半身：胸部以上 / 腰部以上
FRAMING_FULL_BODY = ["full body"]  # 全身：头到脚

# 所有 framing tag（用于冲突检测）
FRAMING_ALL = FRAMING_CLOSE_UP + FRAMING_HALF_BODY + FRAMING_FULL_BODY

# 视角朝向 tag（独立维度，可与 framing 叠加）
VIEWPOINT_TAGS = [
    "from above", "from below", "from side", "from behind",
    "pov", "female pov",
]
