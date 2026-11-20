"""AstrBot 插件：One Last Image 卢浮宫生成器。

把赛璐珞风格的动画截图 / 插画，转换成 One Last Kiss 封面风格的彩色线稿。

渲染算法逐像素移植自 itorr/one-last-image 的 ``html/louvre.js``（MIT 协议），
纯 numpy + Pillow 实现，不需要浏览器。
"""

from __future__ import annotations

import asyncio
import io
import math
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image as ImageComponent
from astrbot.api.message_components import Reply
from astrbot.api.star import Context, Star, StarTools

from . import oli_renderer as renderer

try:  # 兼容不同 AstrBot 版本
    from astrbot.core.config.astrbot_config import AstrBotConfig
except Exception:  # pragma: no cover
    AstrBotConfig = dict  # type: ignore[misc,assignment]

PLUGIN_NAME = "astrbot_plugin_one_last_image"

COMMAND_NAMES = {"卢浮宫", "one_last_image", "onelastimage", "oli"}

HELP_TOKENS = {"帮助", "help", "?", "？", "-h", "--help"}

HELP_TEXT = """🧸 One Last Image · 卢浮宫生成器
把动画截图 / 插画转成 One Last Kiss 封面风格的彩色线稿。

用法：引用一张图，然后发
· /卢浮宫            默认参数生成
· /卢浮宫 精细       线条更细（精细/一般/稍粗/超粗/极粗/浮雕/线稿）
· /卢浮宫 无水印     不叠加右下角光碟水印
· /卢浮宫 初回       换成初回限定盘水印
· /卢浮宫 不排线     去掉暗部铅笔调子
· /卢浮宫 对比       额外输出「效果图 + 原图」对比图

也可以把图片和命令发在同一条消息里。"""


class OneLastImagePlugin(Star):
    """把动画截图 / 插画转成 One Last Kiss 卢浮宫风格线稿。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self.assets = renderer.load_assets(Path(__file__).with_name("assets"))

        self.cache_dir = StarTools.get_data_dir(PLUGIN_NAME) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._semaphore = asyncio.Semaphore(max(1, self._cfg_int("max_concurrent", 2)))
        self._last_cleanup = 0.0
        self._last_used: dict[str, float] = {}

        logger.info(
            "[One Last Image] 已加载，缓存目录：%s（素材 %sx%s）",
            self.cache_dir,
            self.assets.watermark.size[0],
            self.assets.watermark.size[1],
        )

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default):
        try:
            value = self.config.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def _cfg_int(self, key: str, default: int) -> int:
        try:
            return int(self._cfg(key, default))
        except (TypeError, ValueError):
            return default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        value = self._cfg(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "是"}
        return bool(value)

    # ------------------------------------------------------------------
    # 指令
    # ------------------------------------------------------------------

    @filter.command("卢浮宫", alias=set(COMMAND_NAMES) - {"卢浮宫"})
    async def louvre_command(self, event: AstrMessageEvent):
        """把引用的图片转换成 One Last Kiss 封面风格（/卢浮宫 帮助 查看用法）"""
        tokens = self._tokens(event.message_str or "")
        if any(token in HELP_TOKENS for token in tokens):
            yield event.plain_result(HELP_TEXT)
            return

        overrides, compare = self._parse_tokens(tokens)
        async for result in self._run(event, overrides, compare):
            yield result

    @filter.llm_tool(name="one_last_image")
    async def one_last_image_tool(
        self,
        event: AstrMessageEvent,
        compare: bool = False,
        hajimei: bool = False,
        watermark: bool = True,
        line: str = "一般",
    ):
        """把用户这条消息里或引用的图片，转换成 One Last Kiss 封面风格的彩色线稿。

        Args:
            compare(boolean): 是否把原图拼在下方一起输出做对比
            hajimei(boolean): 是否使用初回限定盘样式的右下角水印
            watermark(boolean): 是否叠加右下角光碟水印
            line(string): 线条粗细，可选：精细、一般、稍粗、超粗、极粗、浮雕、线稿
        """
        if not self._cfg_bool("enable_llm_tool", True):
            return
        overrides: dict = {"hajimei": hajimei, "watermark": watermark}
        if line in renderer.CONVOLUTES:
            overrides["convolute"] = line
        async for result in self._run(event, overrides, compare):
            yield result

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    async def _run(self, event: AstrMessageEvent, overrides: dict, compare: bool):
        """取图 -> 渲染 -> 发图。"""
        wait = self._cooldown_left(event)
        if wait > 0:
            yield event.plain_result(f"手速太快啦，{wait:.1f} 秒后再试")
            return

        component = self._find_image(event)
        if component is None:
            yield event.plain_result(
                "没找到图片哦，引用一张图再发 /卢浮宫，"
                "或者把图片和命令发在同一条消息里。",
            )
            return

        try:
            data = await self._read_image(component)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[One Last Image] 读取图片失败：%s", exc)
            yield event.plain_result("这张图我拿不到，换一张试试？")
            return

        max_mb = self._cfg_int("max_input_mb", 20)
        if max_mb > 0 and len(data) > max_mb * 1024 * 1024:
            yield event.plain_result(f"图片太大了（超过 {max_mb} MB），先压一下再发吧。")
            return

        options = self._build_options(overrides)
        started = time.perf_counter()
        try:
            async with self._semaphore:
                path = await asyncio.to_thread(self._render_sync, data, options, compare)
        except Exception as exc:  # noqa: BLE001
            logger.error("[One Last Image] 渲染失败：%s", exc, exc_info=True)
            yield event.plain_result("生成失败了，换张图或者稍后再试。")
            return

        logger.info(
            "[One Last Image] 生成完成 %.0f ms %s",
            (time.perf_counter() - started) * 1000,
            path.name,
        )
        yield event.image_result(str(path))

    def _render_sync(self, data: bytes, options: renderer.RenderOptions, compare: bool) -> Path:
        """在子线程里完成解码 + 渲染 + 存盘。"""
        with Image.open(io.BytesIO(data)) as raw:
            raw.load()
            source = self._prepare_source(ImageOps.exif_transpose(raw))
            output = renderer.render(source, options, self.assets)
            if compare:
                output = self._stack_compare(output, source)
            path = self.cache_dir / f"oli-{uuid.uuid4().hex[:12]}.jpg"
            output.save(
                path,
                "JPEG",
                quality=max(50, min(98, self._cfg_int("jpeg_quality", 92))),
                subsampling=1,
                optimize=True,
            )
        self._cleanup_cache()
        return path

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _tokens(self, text: str) -> list[str]:
        return [token for token in text.replace("/", " ").split() if token not in COMMAND_NAMES]

    def _parse_tokens(self, tokens: list[str]) -> tuple[dict, bool]:
        """把命令后的自然语言选项翻译成渲染参数。"""
        overrides: dict = {}
        compare = False
        for token in tokens:
            if token in renderer.CONVOLUTES:
                overrides["convolute"] = token
            elif token in {"无水印", "不加水印", "nowatermark"}:
                overrides["watermark"] = False
            elif token in {"有水印", "加水印", "watermark"}:
                overrides["watermark"] = True
            elif token in {"初回", "初回限定", "hajimei"}:
                overrides["hajimei"] = True
            elif token in {"普通盘", "普通版", "normal"}:
                overrides["hajimei"] = False
            elif token in {"不排线", "无排线", "noshade"}:
                overrides["shade"] = False
            elif token in {"排线", "开排线", "shade"}:
                overrides["shade"] = True
            elif token in {"不降噪", "nodenoise", "no-denoise"}:
                overrides["denoise"] = False
            elif token in {"降噪", "denoise"}:
                overrides["denoise"] = True
            elif token in {"不走彩虹", "无渐变", "nokuma"}:
                overrides["kuma"] = False
            elif token in {"对比", "对比图", "compare", "diff"}:
                compare = True
        return overrides, compare

    def _build_options(self, overrides: dict) -> renderer.RenderOptions:
        convolute = str(self._cfg("convolute", "一般"))
        params = {
            "watermark": self._cfg_bool("watermark", True),
            "hajimei": self._cfg_bool("hajimei", False),
            "shade": self._cfg_bool("shade", True),
            "shade_limit": self._cfg_int("shade_limit", 108),
            "shade_light": self._cfg_int("shade_light", 80),
            "denoise": self._cfg_bool("denoise", True),
            "light": self._cfg_int("light", 0),
            "light_cut": self._cfg_int("light_cut", 128),
            "dark_cut": self._cfg_int("dark_cut", 118),
            "kuma": self._cfg_bool("kuma", True),
            "convolute": convolute if convolute in renderer.CONVOLUTES else "一般",
        }
        params.update(overrides)
        if params["convolute"] not in renderer.CONVOLUTES:
            params["convolute"] = "一般"
        return renderer.RenderOptions(**params)

    def _find_image(self, event: AstrMessageEvent) -> ImageComponent | None:
        """优先取引用消息里的图，其次取本条消息里的图。"""
        messages = list(event.get_messages())
        for component in messages:
            if isinstance(component, Reply):
                for sub in component.chain or []:
                    if isinstance(sub, ImageComponent):
                        return sub
        for component in messages:
            if isinstance(component, ImageComponent):
                return component
        return None

    async def _read_image(self, component: ImageComponent) -> bytes:
        path = await component.convert_to_file_path()
        return await asyncio.to_thread(Path(path).read_bytes)

    def _prepare_source(self, image: Image.Image) -> Image.Image:
        """铺白透明背景、限制最大像素与宽度。"""
        if image.mode in {"RGBA", "LA", "P", "PA"}:
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, (0, 0), image)
            image = background
        else:
            image = image.convert("RGB")

        limit = self._cfg_int("max_input_pixels", 24_000_000)
        if limit > 0 and image.width * image.height > limit:
            scale = math.sqrt(limit / (image.width * image.height))
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )

        max_width = self._cfg_int("max_width", 1600)
        if max_width > 0 and image.width > max_width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def _stack_compare(output: Image.Image, source: Image.Image) -> Image.Image:
        """上效果图、下原图，拼成一张对比图。"""
        width, height = output.size
        canvas = Image.new("RGB", (width, height * 2), (255, 255, 255))
        canvas.paste(output, (0, 0))
        canvas.paste(source.resize((width, height), Image.Resampling.LANCZOS), (0, height))
        return canvas

    def _cooldown_left(self, event: AstrMessageEvent) -> float:
        cooldown = self._cfg_int("cooldown_seconds", 0)
        if cooldown <= 0:
            return 0.0
        user = event.get_sender_id() or event.get_session_id() or "anon"
        now = time.monotonic()
        last = self._last_used.get(user, 0.0)
        if now - last < cooldown:
            return cooldown - (now - last)
        self._last_used[user] = now
        return 0.0

    def _cleanup_cache(self, force: bool = False) -> None:
        """按 TTL 清理历史输出，避免缓存目录无限增长。"""
        now = time.time()
        if not force and now - self._last_cleanup < 600:
            return
        self._last_cleanup = now
        ttl = self._cfg_int("cache_ttl_minutes", 60) * 60
        if ttl <= 0:
            return
        for path in self.cache_dir.glob("oli-*.jpg"):
            try:
                if now - path.stat().st_mtime > ttl:
                    path.unlink()
            except OSError:
                pass
