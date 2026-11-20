"""开发用冒烟测试：不连 QQ，直接构造一个假的 AstrMessageEvent 跑完整链路。

需要 AstrBot 源码可导入（用它自带的 Python 跑最省事）：

    python tools/smoke_test.py                 # 用内置合成图测试
    python tools/smoke_test.py 某张图.jpg       # 用指定图片测试

AstrBot 源码目录默认按常见位置查找，也可以用环境变量指定：

    set ASTRBOT_PATH=F:\\AstrBot\\backend\\app
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))


def find_astrbot() -> Path | None:
    """定位 AstrBot 源码目录（含 astrbot 包的那一层）。"""
    candidates = []
    if os.environ.get("ASTRBOT_PATH"):
        candidates.append(Path(os.environ["ASTRBOT_PATH"]))
    candidates += [
        Path.home() / "AstrBot" / "backend" / "app",
        Path.home() / "astrbot",
        Path.cwd(),
        ROOT.parent / "AstrBot" / "backend" / "app",
        Path(r"F:\AstrBot\backend\app"),
    ]
    for path in candidates:
        if (path / "astrbot" / "__init__.py").exists():
            return path
    return None


ASTRBOT_APP = find_astrbot()
if ASTRBOT_APP is None:
    print(
        "找不到 AstrBot 源码目录。\n"
        "请设置环境变量 ASTRBOT_PATH 指向 AstrBot 的 backend/app 目录后重试。",
    )
    raise SystemExit(2)
sys.path.insert(0, str(ASTRBOT_APP))

from astrbot.core.message.components import Image as ImageComponent  # noqa: E402
from astrbot.core.message.components import Plain, Reply  # noqa: E402
from astrbot.core.platform.astr_message_event import AstrMessageEvent  # noqa: E402
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember  # noqa: E402
from astrbot.core.platform.message_type import MessageType  # noqa: E402
from astrbot.core.platform.platform_metadata import PlatformMetadata  # noqa: E402

import importlib  # noqa: E402

plugin_main = importlib.import_module(f"{ROOT.name}.main")

PLATFORM = PlatformMetadata(name="aiocqhttp", description="test", id="aiocqhttp")


def make_sample() -> Path:
    """没有指定图片时，画一张渐变 + 深色块的合成图来测试。"""
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = 640, 400
    x = np.linspace(0, 1, width)[None, :, None]
    y = np.linspace(0, 1, height)[:, None, None]
    base = np.concatenate(
        [
            (0.2 + 0.8 * x) * np.ones((height, 1, 1)),
            (0.3 + 0.6 * y) * np.ones((1, width, 1)),
            0.5 * np.ones((height, width, 1)),
        ],
        axis=2,
    )
    img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 60, 260, 260), fill=(20, 20, 30))
    draw.rectangle((320, 120, 560, 300), fill=(230, 240, 255), outline=(10, 10, 10), width=6)
    draw.line((40, 360, 600, 320), fill=(5, 5, 5), width=5)
    path = Path(tempfile.gettempdir()) / "oli-smoke-sample.jpg"
    img.save(path, quality=95)
    return path


class FakeContext:
    def get_config(self, umo=None):
        return None


def make_event(
    text: str,
    *,
    image: Path | None = None,
    reply_image: Path | None = None,
) -> AstrMessageEvent:
    components = []
    if reply_image is not None:
        components.append(
            Reply(id="1", chain=[ImageComponent.fromFileSystem(reply_image)])
        )
    if image is not None:
        components.append(ImageComponent.fromFileSystem(image))
    components.append(Plain(text=text))

    abm = AstrBotMessage()
    abm.type = MessageType.GROUP_MESSAGE
    abm.self_id = "10000"
    abm.session_id = "20000"
    abm.message_id = "m1"
    abm.group_id = "20000"
    abm.sender = MessageMember(user_id="30000", nickname="tester")
    abm.message = components
    abm.message_str = text
    abm.raw_message = None

    return AstrMessageEvent(
        message_str=text,
        message_obj=abm,
        platform_meta=PLATFORM,
        session_id="20000",
    )


def component_path(comp: ImageComponent) -> Path | None:
    if comp.path:
        return Path(comp.path)
    file = comp.file or ""
    if file.startswith("file://"):
        return Path(unquote(urlparse(file).path.lstrip("/")))
    return None


def describe(results) -> str:
    parts = []
    for result in results:
        for comp in getattr(result, "chain", None) or []:
            if isinstance(comp, Plain):
                parts.append(f"text({comp.text[:36]})")
            elif isinstance(comp, ImageComponent):
                path = component_path(comp)
                parts.append(f"image({path.name if path else comp.file})")
            else:
                parts.append(type(comp).__name__)
    return ", ".join(parts) or "<empty>"


async def main(argv: list[str]) -> int:
    sample = Path(argv[1]).resolve() if len(argv) > 1 else make_sample()
    if not sample.exists():
        print(f"图片不存在: {sample}")
        return 2
    print(f"AstrBot: {ASTRBOT_APP}")
    print(f"测试图: {sample}")

    plugin = plugin_main.OneLastImagePlugin(FakeContext(), config={})
    failures = 0

    checks = [
        ("引用图片 + 默认参数", make_event("卢浮宫", reply_image=sample), True),
        ("同条消息里的图片", make_event("卢浮宫", image=sample), True),
        ("带选项", make_event("卢浮宫 精细 无水印 初回", reply_image=sample), True),
        ("对比图", make_event("卢浮宫 对比", reply_image=sample), True),
        ("线稿方案", make_event("卢浮宫 线稿", reply_image=sample), True),
        ("未知选项仍能出图", make_event("卢浮宫 随便说点什么", reply_image=sample), True),
        ("帮助", make_event("卢浮宫 帮助"), False),
        ("没有图片", make_event("卢浮宫"), False),
    ]

    for label, event, needs_image in checks:
        results = []
        async for result in plugin.louvre_command(event):
            results.append(result)
        print(f"[{label}] -> {describe(results)}")
        if not results:
            print("   !! 没有任何输出")
            failures += 1
            continue
        if not needs_image:
            continue
        images = [
            comp
            for result in results
            for comp in (getattr(result, "chain", None) or [])
            if isinstance(comp, ImageComponent)
        ]
        if not images:
            print("   !! 预期有图片输出")
            failures += 1
            continue
        for comp in images:
            path = component_path(comp)
            if path is None or not path.exists():
                print(f"   !! 文件不存在: {path or comp.file}")
                failures += 1
                continue
            from PIL import Image as PILImage

            with PILImage.open(path) as im:
                print(
                    f"   -> {path.name} {im.size} {im.mode} "
                    f"{path.stat().st_size // 1024} KB",
                )

    overrides, compare = plugin._parse_tokens(["极粗", "无水印", "初回", "不排线", "对比"])
    expected = {"convolute": "极粗", "watermark": False, "hajimei": True, "shade": False}
    print(f"[解析] overrides={overrides} compare={compare}")
    if overrides != expected or not compare:
        print("   !! 选项解析不符合预期")
        failures += 1

    print("\n结果:", "PASS" if failures == 0 else f"{failures} 处失败")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
