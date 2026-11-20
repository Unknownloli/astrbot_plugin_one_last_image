"""One Last Image 渲染核心。

本模块是 itorr/one-last-image 中 ``html/louvre.js`` 的 Python 移植版，
用 NumPy + Pillow 复现浏览器 Canvas 上的同一套像素管线：

    裁切/缩放 -> 灰度 -> 铅笔排线(暗部) -> 亮度 -> 降噪 -> 卷积(线稿)
    -> 明暗对比裁切 -> 彩虹渐变上色(alpha=暗度) -> 缩小再放大 -> 水印 -> 白底合成

设计原则：
- 不依赖 AstrBot / 浏览器 / Node，可独立单元测试。
- 与 louvre.js 的数值行为逐段对齐（含 Uint8ClampedArray 的截断与四舍五入、
  createLinearGradient 的投影方式、drawImage 的双线性缩放）。
- 纯 CPU 计算，800~1920 宽的图通常在数百毫秒内完成。

原作者：itorr <https://github.com/itorr/one-last-image>  (MIT License)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

__all__ = [
    "CONVOLUTES",
    "CONVOLUTE_NAMES",
    "Assets",
    "RenderOptions",
    "load_assets",
    "render",
    "target_size",
]

# ---------------------------------------------------------------------------
# 卷积核（对应 document.js 里的 Convolutes）
# ---------------------------------------------------------------------------


def _average(side: int) -> list[float]:
    return [1.0 / (side * side)] * (side * side)


CONVOLUTES: dict[str, list[float] | None] = {
    "精细": _average(5),
    "一般": _average(7),
    "稍粗": _average(9),
    "超粗": _average(11),
    "极粗": _average(13),
    "浮雕": [1, 1, 1, 1, 1, -1, -1, -1, -1],
    "线稿": None,  # 原版里为 null：不做卷积、直接输出灰度
}

CONVOLUTE_NAMES: tuple[str, ...] = tuple(CONVOLUTES.keys())

# 渐变停靠点（对应 louvre.js 中 config.kuma 分支的 addColorStop）
_GRADIENT_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (0xFB, 0xBA, 0x30)),
    (0.4, (0xFC, 0x72, 0x35)),
    (0.6, (0xFC, 0x35, 0x4E)),
    (0.7, (0xCF, 0x36, 0xDF)),
    (0.8, (0x37, 0xB5, 0xD9)),
    (1.0, (0x3E, 0xB6, 0xDA)),
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderOptions:
    """一次渲染的开关与参数，字段与原版 ``style`` 对象一一对应。"""

    zoom: float = 1.0
    """缩放系数，1 表示原始分辨率（再受 max_width 限制）。"""

    cover: bool = False
    """是否裁成正方形（原版 ``config.cover``，网页版未开放）。"""

    max_width: int = 1920
    """输出最大宽度，对应 louvre.js 里的 ``maxWidth``。"""

    # 铅笔排线
    shade: bool = True
    shade_limit: int = 108
    """亮度低于该值的区域才排调子。"""
    shade_light: int = 80
    """调子深度。"""

    # 亮度
    light: int = 0

    # 降噪
    denoise: bool = True

    # 卷积
    convolute: str = "一般"
    convolute_diff: bool = True
    """True = 原图减模糊（USM，出线稿）；False = 直接输出模糊图。"""
    convolute2: str | None = None

    # 明暗裁切
    light_cut: int = 128
    dark_cut: int = 118

    # 彩虹渐变
    kuma: bool = True

    # 水印
    watermark: bool = True
    hajimei: bool = False
    """True = 使用初回限定盘那半张水印。"""

    def with_(self, **kwargs) -> "RenderOptions":
        """返回替换了部分字段的新配置（本类不可变）。"""
        unknown = set(kwargs) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"未知参数: {', '.join(sorted(unknown))}")
        return replace(self, **kwargs)


@dataclass(frozen=True)
class Assets:
    """渲染所需的静态素材。"""

    pencil_texture: np.ndarray
    """铅笔纹理的 R 通道，shape=(1200, 1200)，float64，0~255。"""
    watermark: Image.Image
    """水印图（RGBA，1200x800，上下两半分别是普通盘/初回限定盘）。"""


def load_assets(asset_dir: str | Path | None = None) -> Assets:
    """载入 ``assets/`` 下的铅笔纹理与水印。"""
    base = Path(asset_dir) if asset_dir else Path(__file__).with_name("assets")
    texture_path = base / "pencil-texture.jpg"
    logo_path = base / "one-last-image-logo2.png"
    for p in (texture_path, logo_path):
        if not p.exists():
            raise FileNotFoundError(f"缺少素材文件: {p}")

    with Image.open(texture_path) as tex:
        # 原版只读取 pencilTexturePixel.data[i]，即 R 通道
        pencil = np.asarray(tex.convert("RGB"), dtype=np.float64)[..., 0].copy()

    with Image.open(logo_path) as logo:
        watermark = logo.convert("RGBA")

    return Assets(pencil_texture=pencil, watermark=watermark)


# ---------------------------------------------------------------------------
# 基础数值工具
# ---------------------------------------------------------------------------


def _clamp_u8(a: np.ndarray) -> np.ndarray:
    """模拟 Uint8ClampedArray 写入：截断到 0~255 并四舍五入。"""
    return np.rint(np.clip(a, 0.0, 255.0))


def _js_round(x: float) -> int:
    """JS 的 Math.round：.5 一律向上（Python 的 round 是银行家舍入）。"""
    return int(math.floor(x + 0.5))


def _box_blur(a: np.ndarray, side: int) -> np.ndarray:
    """side x side 均值滤波，边界按边缘复制（对齐 JS 里的 clamp 索引）。"""
    if side <= 1:
        return a.copy()
    radius = side // 2
    padded = np.pad(a, ((radius, radius), (radius, radius)), mode="edge")
    integral = padded.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    total = (
        integral[side:, side:]
        - integral[:-side, side:]
        - integral[side:, :-side]
        + integral[:-side, :-side]
    )
    return total / float(side * side)


def _convolution(a: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    """通用卷积；weights 为 side*side 扁平数组，边界同样复制边缘。"""
    w = list(weights)
    side = int(round(math.sqrt(len(w))))
    radius = side // 2
    h, width = a.shape
    padded = np.pad(a, ((radius, radius), (radius, radius)), mode="edge")
    out = np.zeros_like(a, dtype=np.float64)
    for cy in range(side):
        for cx in range(side):
            weight = w[cy * side + cx]
            if weight:
                out += weight * padded[cy : cy + h, cx : cx + width]
    return out


def _is_average(weights: Iterable[float]) -> bool:
    w = list(weights)
    if not w:
        return False
    expected = 1.0 / len(w)
    return all(abs(x - expected) < 1e-12 for x in w)


def _convolve_y(a: np.ndarray, weights: Iterable[float] | None) -> np.ndarray:
    """对应 louvre.js 的 convoluteY：只处理单通道灰度。"""
    if weights is None:
        return a.copy()
    w = list(weights)
    side = int(round(math.sqrt(len(w))))
    if _is_average(w):
        return _box_blur(a, side)
    return _convolution(a, w)


def _resample_axis(arr: np.ndarray, new_len: int, axis: int, span: float | None = None) -> np.ndarray:
    """沿单个轴做双线性重采样（像素中心对齐、边界复制）。

    ``span`` 为目标矩形长度，可与 ``new_len`` 不同：浏览器按浮点目标矩形
    拉伸后，被画布边界裁掉的正是这种情形（例如高度 1139.2 只保留 1139 行）。
    """
    src = arr.shape[axis]
    span = float(new_len) if not span else float(span)
    if src == new_len and abs(span - new_len) < 1e-9:
        return arr
    pos = (np.arange(new_len, dtype=np.float64) + 0.5) * (src / span) - 0.5
    i0 = np.floor(pos).astype(np.int64)
    frac = (pos - i0).astype(np.float32)
    i0c = np.clip(i0, 0, src - 1)
    i1c = np.clip(i0 + 1, 0, src - 1)
    a0 = np.take(arr, i0c, axis=axis)
    a1 = np.take(arr, i1c, axis=axis)
    shape = [1] * arr.ndim
    shape[axis] = new_len
    return a0 + (a1 - a0) * frac.reshape(shape)


def _bilinear(
    a: np.ndarray,
    new_w: int,
    new_h: int,
    span_w: float | None = None,
    span_h: float | None = None,
) -> np.ndarray:
    """双线性缩放（可分离两遍实现），像素中心对齐，边界复制。

    与 Canvas ``drawImage`` 在 0.5~2 倍区间内的采样行为一致；
    多通道（末尾维度）一次处理。
    """
    new_w = max(1, int(new_w))
    new_h = max(1, int(new_h))
    src_h, src_w = a.shape[:2]
    if (
        src_w == new_w
        and src_h == new_h
        and (span_w is None or abs(span_w - new_w) < 1e-9)
        and (span_h is None or abs(span_h - new_h) < 1e-9)
    ):
        return np.asarray(a, dtype=np.float64).copy()

    arr = np.ascontiguousarray(a, dtype=np.float32)
    out = _resample_axis(arr, new_w, 1, span_w)
    out = _resample_axis(out, new_h, 0, span_h)
    return out.astype(np.float64)


def _resize_rgba_premultiplied(a: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """按预乘 alpha 缩放 RGBA 浮点图，等价于 Canvas 的合成式缩放。"""
    alpha = a[..., 3:4] / 255.0
    premultiplied = np.concatenate([a[..., :3] * alpha, a[..., 3:4]], axis=2)
    resized = _bilinear(premultiplied, new_w, new_h)
    out_alpha = resized[..., 3:4]
    safe = np.maximum(out_alpha, 1e-6)
    rgb = resized[..., :3] * 255.0 / safe
    return np.concatenate([np.clip(rgb, 0.0, 255.0), out_alpha], axis=2)


def _gradient(width: int, height: int) -> np.ndarray:
    """复现 createLinearGradient(0,0,w,h) 沿对角线投影的取色。"""
    xs = np.arange(width, dtype=np.float64) + 0.5
    ys = np.arange(height, dtype=np.float64) + 0.5
    denominator = float(width * width + height * height)
    t = (xs[None, :] * width + ys[:, None] * height) / denominator
    np.clip(t, 0.0, 1.0, out=t)

    stops = np.array([s[0] for s in _GRADIENT_STOPS])
    colors = np.array([s[1] for s in _GRADIENT_STOPS], dtype=np.float64)
    out = np.empty((height, width, 3), dtype=np.float64)
    for ch in range(3):
        out[..., ch] = np.interp(t, stops, colors[:, ch])
    return np.rint(out)


# ---------------------------------------------------------------------------
# 尺寸计算
# ---------------------------------------------------------------------------


def target_size(width: int, height: int, options: RenderOptions) -> tuple[int, int]:
    """返回输出画布尺寸（与 louvre.js 完全一致，含浮点截断）。"""
    out_w, out_h, _, _ = _layout(width, height, options)
    return out_w, out_h


def _layout(
    width: int, height: int, options: RenderOptions
) -> tuple[int, int, float, float]:
    """返回 (画布宽, 画布高, 目标矩形宽, 目标矩形高)。

    末两项是 ``drawImage`` 用的浮点尺寸：canvas.width/height 会被截断，
    而绘制用的是未截断的浮点值，两者可能差一点点。
    """
    zoom = options.zoom if options.zoom and options.zoom > 0 else 1.0
    dest_w = float(_js_round(width / zoom))
    dest_h = float(_js_round(height / zoom))

    max_width = options.max_width
    if max_width and dest_w > max_width:
        dest_h = dest_h * max_width / dest_w
        dest_w = float(max_width)

    if options.cover:
        orientation = width / height
        if orientation > 1:
            dest_w = dest_h
        else:
            dest_h = dest_w

    return max(1, int(dest_w)), max(1, int(dest_h)), dest_w, dest_h


def _source_box(
    width: int, height: int, options: RenderOptions
) -> tuple[int, int, int, int]:
    """返回 ``drawImage`` 的源矩形 (left, top, w, h)。"""
    if not options.cover:
        return 0, 0, width, height
    orientation = width / height
    if orientation > 1:
        cut_left = _js_round((orientation - 1) * height / 2)
        return cut_left, 0, height, height
    cut_top = _js_round((1 - orientation) * height / 2)
    return 0, cut_top, width, width


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _resize_canvas(
    img: Image.Image, out_w: int, out_h: int, dest_w: float, dest_h: float
) -> Image.Image:
    """复刻 Canvas ``drawImage`` 的采样行为。

    缩放比小于 0.5 时浏览器会走 mipmap（按 2 的幂做面积平均再双线性），
    其余情况直接双线性采样；目标矩形是浮点尺寸，超出画布的部分被裁掉。
    """
    cur = img
    while cur.width > dest_w and dest_w / cur.width < 0.5:
        nxt = (
            max(1, int(round(cur.width / 2))),
            max(1, int(round(cur.height / 2))),
        )
        if nxt == cur.size:
            break
        cur = cur.resize(nxt, Image.Resampling.BOX)
    if cur.size == (out_w, out_h) and (dest_w, dest_h) == (out_w, out_h):
        return cur
    arr = np.asarray(cur, dtype=np.float64)
    out = _bilinear(arr, out_w, out_h, dest_w, dest_h)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def _prepare_canvas(
    source: Image.Image, options: RenderOptions
) -> np.ndarray:
    """裁切 + 缩放 + 灰度，对应 canvas/setImageAndDraw 的前半段。"""
    img = ImageOps.exif_transpose(source).convert("RGB")
    width, height = img.size
    out_w, out_h, dest_w, dest_h = _layout(width, height, options)
    box = _source_box(width, height, options)

    crop = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
    crop = _resize_canvas(crop, out_w, out_h, dest_w, dest_h)

    rgb = np.asarray(crop, dtype=np.float64)
    # 浏览器：y = Math.floor(r*.299 + g*.587 + b*.114)
    return np.floor(rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)


def _pencil_shade(
    gray: np.ndarray, options: RenderOptions, assets: Assets
) -> np.ndarray:
    """暗部排铅笔调子，返回可作为 alpha 下限的 0~255 强度图。"""
    height, width = gray.shape
    side = max(width, height)

    # 纹理被拉伸成 max(w,h) 的正方形，再裁出左上角 w x h
    texture = _bilinear(assets.pencil_texture, side, side)[:height, :width]
    if texture.shape != gray.shape:
        texture = np.pad(
            texture,
            ((0, height - texture.shape[0]), (0, width - texture.shape[1])),
            mode="edge",
        )

    # 暗部为 255、亮部为 0；原版先画到 1/4 大小再放大回来，形成柔和的调子
    mask = np.where(gray <= options.shade_limit, 255.0, 0.0)
    mask = _bilinear(mask, max(1, width // 4), max(1, height // 4))
    mask = _bilinear(mask, width, height)

    return _clamp_u8((255.0 - texture) / 255.0 * mask / 255.0 * options.shade_light)


def render(
    source: Image.Image,
    options: RenderOptions | None = None,
    assets: Assets | None = None,
) -> Image.Image:
    """把一张图渲染成 One Last Kiss 封面风格，返回 RGB 图。

    Args:
        source: 输入图（任意 PIL 可解码的图，会自动应用 EXIF 旋转）。
        options: 渲染参数，默认即网页版默认效果。
        assets: 素材；缺省时从包内 ``assets/`` 载入。

    Returns:
        RGB 模式的 PIL 图像。
    """
    opts = options or RenderOptions()
    if assets is None:
        assets = load_assets()

    if opts.convolute not in CONVOLUTES:
        raise ValueError(
            f"未知线条方案 {opts.convolute!r}，可选：{', '.join(CONVOLUTE_NAMES)}"
        )
    if opts.convolute2 is not None and opts.convolute2 not in CONVOLUTES:
        raise ValueError(f"未知线条方案 {opts.convolute2!r}")

    gray = _prepare_canvas(source, opts)
    out_h, out_w = gray.shape

    shade = None
    if opts.shade:
        shade = _pencil_shade(gray, opts, assets)

    if opts.light:
        gray = _clamp_u8(gray + gray * (opts.light / 100.0))

    if opts.denoise:
        gray = _clamp_u8(_box_blur(gray, 3))

    matrix = CONVOLUTES[opts.convolute]
    if matrix is not None:
        blurred = _clamp_u8(_convolve_y(gray, matrix))
        if opts.convolute_diff:
            if opts.convolute2 is None:
                base = gray
            else:
                base = _clamp_u8(_convolve_y(gray, CONVOLUTES[opts.convolute2]))
            lum = _clamp_u8(128.0 + base - blurred)
        else:
            lum = blurred
    else:
        lum = gray.copy()

    if matrix is not None and (opts.light_cut or opts.dark_cut):
        scale = 255.0 / (255.0 - opts.light_cut - opts.dark_cut)
        lum = _clamp_u8(np.maximum(0.0, (lum - opts.dark_cut) * scale))

    if opts.kuma:
        rgb = _gradient(out_w, out_h)
        alpha = np.clip(255.0 - lum, 0.0, 255.0)
        if shade is not None:
            alpha = np.maximum(alpha, shade)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    else:
        gray_rgb = np.repeat(lum[..., None], 3, axis=2)
        rgba = np.concatenate([gray_rgb, np.full((out_h, out_w, 1), 255.0)], axis=2)

    # 缩小再放大，制造"印上去"的柔化感
    min_w = max(1, int(out_w / 1.4))
    min_h = max(1, int(out_h / 1.3))
    rgba = _resize_rgba_premultiplied(rgba, min_w, min_h)
    rgba = _resize_rgba_premultiplied(rgba, out_w, out_h)

    canvas = Image.fromarray(np.clip(np.rint(rgba), 0, 255).astype(np.uint8), "RGBA")

    if opts.watermark:
        canvas = _draw_watermark(canvas, opts, assets)

    background = Image.new("RGB", canvas.size, (255, 255, 255))
    background.paste(canvas, (0, 0), canvas)
    return background


def _sample_offset(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """把图像整体平移 (dx, dy) 像素采样回来，边界复制。"""
    height, width = a.shape[:2]
    flat = np.ascontiguousarray(a).reshape(height, width, -1).astype(np.float64)
    ys = np.arange(height) - dy
    xs = np.arange(width) - dx
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = (ys - y0).reshape(-1, 1, 1)
    fx = (xs - x0).reshape(1, -1, 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x0 + 1, 0, width - 1)
    p00 = flat[y0c][:, x0c]
    p01 = flat[y0c][:, x1c]
    p10 = flat[y1c][:, x0c]
    p11 = flat[y1c][:, x1c]
    top = p00 + (p01 - p00) * fx
    bottom = p10 + (p11 - p10) * fx
    out = top + (bottom - top) * fy
    return out.reshape(a.shape)


def _draw_watermark(
    canvas: Image.Image, options: RenderOptions, assets: Assets
) -> Image.Image:
    """右下角叠加 One Last Kiss 光碟封面水印。

    位置与尺寸完全照搬 louvre.js（含浮点坐标），并按浏览器的方式做
    预乘 alpha 合成与亚像素落点，保证与线上效果一致。
    """
    width, height = canvas.size
    logo = assets.watermark
    logo_w, logo_h = logo.size
    half_h = logo_h // 2

    set_w = width * 0.3
    set_h = set_w / logo_w * half_h
    if width / height > 1.1:
        set_h = height * 0.15
        set_w = set_h / half_h * logo_w

    cut_top = half_h if options.hajimei else 0
    half = logo.crop((0, cut_top, logo_w, cut_top + half_h))

    target_w = max(1, int(round(set_w)))
    target_h = max(1, int(round(set_h)))
    factor = target_w / logo_w
    sampler = Image.Resampling.BOX if factor < 0.5 else Image.Resampling.BILINEAR
    half = half.resize((target_w, target_h), sampler)

    left = width - set_w - set_h * 0.2
    top = height - set_h - set_h * 0.16
    left = min(max(0.0, left), max(0.0, width - target_w))
    top = min(max(0.0, top), max(0.0, height - target_h))

    ix, iy = int(math.floor(left)), int(math.floor(top))
    fx, fy = left - ix, top - iy
    x1 = min(width, ix + target_w)
    y1 = min(height, iy + target_h)
    if x1 <= ix or y1 <= iy:
        return canvas

    # 预乘 alpha 的 logo 图层（只处理可见的那一小块）
    logo_rgba = np.asarray(half, dtype=np.float64) / 255.0
    logo_alpha = logo_rgba[..., 3:4]
    logo_pm = np.concatenate([logo_rgba[..., :3] * logo_alpha, logo_alpha], axis=2)
    if fx or fy:
        logo_pm = _sample_offset(logo_pm, fx, fy)
    logo_pm = logo_pm[: y1 - iy, : x1 - ix]

    region = np.asarray(canvas.crop((ix, iy, x1, y1)), dtype=np.float64) / 255.0
    base_alpha = region[..., 3:4]
    base_pm = np.concatenate([region[..., :3] * base_alpha, base_alpha], axis=2)

    src_a = logo_pm[..., 3:4]
    out_pm = logo_pm + base_pm * (1.0 - src_a)
    out_a = np.clip(out_pm[..., 3:4], 0.0, 1.0)
    out_rgb = out_pm[..., :3] / np.maximum(out_a, 1e-6)
    out = np.concatenate([np.clip(out_rgb, 0.0, 1.0), out_a], axis=2)

    result = canvas.copy()
    result.paste(
        Image.fromarray(np.rint(out * 255.0).astype(np.uint8), "RGBA"), (ix, iy)
    )
    return result
