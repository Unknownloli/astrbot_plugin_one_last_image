# One Last Image · 卢浮宫生成器

> 把**赛璐珞风格的动画截图 / 插画**，转换成《One Last Kiss》封面风格的彩色线稿。

AstrBot 插件版 · 作者 [@Unknownloli](https://github.com/Unknownloli)
· [问题反馈 / 建议](https://github.com/Unknownloli/astrbot_plugin_one_last_image/issues)

渲染算法逐像素移植自 [itorr/one-last-image](https://github.com/itorr/one-last-image)
的 `html/louvre.js`，纯 **numpy + Pillow** 实现，不需要浏览器、Node 或联网渲染。

## 效果

引用一张图，发 `/卢浮宫`，机器人回一张同款封面：

| 输入 | 输出 |
| --- | --- |
| 动画截图 / 插画 | 白底、彩色线条、右下角圆盘水印 |

- 默认参数下与官方网页版**逐像素对齐**（平均误差约 0.4/255，99 分位 ≤4）
- 1377×817 的图约 0.7 秒出图（纯 CPU）

## 安装

**方式一：上传压缩包（推荐）**

1. 到 [Releases](https://github.com/Unknownloli/astrbot_plugin_one_last_image/releases)
   下载 `astrbot_plugin_one_last_image-v1.0.0.zip`
2. AstrBot WebUI →「插件」→ 从文件安装 → 选中这个 zip
3. 重载 / 启用插件

**方式二：手动放目录**

把整个 `astrbot_plugin_one_last_image` 目录放进 AstrBot 的插件目录：

- 源码部署：`data/plugins/`
- 桌面版：`~/.astrbot/data/plugins/`

然后到 WebUI 的「插件」页重载或启用。

依赖 `pillow` 与 `numpy`，AstrBot 自带，无需额外安装。

> 插件的 `metadata.yaml` 已填好本仓库地址，之后可以直接在插件页检查更新。


## 用法

```
/卢浮宫                    # 用默认参数生成
/卢浮宫 精细               # 线条更细
/卢浮宫 极粗               # 线条更粗
/卢浮宫 无水印             # 不要右下角光碟水印
/卢浮宫 初回               # 换成初回限定盘水印
/卢浮宫 不排线             # 去掉暗部铅笔调子
/卢浮宫 对比               # 输出「效果图 + 原图」对比图
/卢浮宫 帮助               # 查看用法
```

取图顺序：

1. **引用/回复**一张图，再发命令（推荐）
2. 把图片和命令发在**同一条消息**里

线条方案：`精细` `一般` `稍粗` `超粗` `极粗` `浮雕` `线稿`
（`线稿` 不做卷积，直接输出灰度；`浮雕` 是原版的彩蛋卷积核）

也可以不敲命令，直接说「把这张图变成卢浮宫风格」——插件注册了同名大模型工具，
由模型自行调用（可在配置里关掉）。

## 配置

WebUI 插件配置页可调，全部有默认值：

| 键 | 说明 | 默认 |
| --- | --- | --- |
| `watermark` | 叠加右下角光碟水印 | `true` |
| `hajimei` | 默认用初回限定盘水印 | `false` |
| `convolute` | 线条方案 | `一般` |
| `shade` / `shade_light` / `shade_limit` | 暗部铅笔调子的开关、深度、阈值 | `true` / `80` / `108` |
| `denoise` | 先做 3×3 降噪 | `true` |
| `light` | 整体亮度 | `0` |
| `light_cut` / `dark_cut` | 明暗裁切（决定黑白对比度） | `128` / `118` |
| `kuma` | 彩虹渐变上色 | `true` |
| `max_width` | 输入图最大宽度，越小越快 | `1600` |
| `max_input_mb` / `max_input_pixels` | 输入体积/像素上限 | `20` / `2400 万` |
| `cooldown_seconds` | 每人冷却，0 为不限制 | `0` |
| `max_concurrent` | 同时渲染任务数 | `2` |
| `cache_ttl_minutes` | 生成结果保留时间 | `60` |
| `enable_llm_tool` | 允许大模型主动调用 | `true` |

## 开发

`tools/smoke_test.py` 不连 QQ，直接构造假事件跑完整链路（取图 → 渲染 → 出图）：

```bash
# 用 AstrBot 自带的 Python 跑
python tools/smoke_test.py                      # 内置合成图，开箱即用
python tools/smoke_test.py path/to/sample.jpg   # 指定测试图
```

找不到 AstrBot 源码时，用 `ASTRBOT_PATH` 指向它的 `backend/app` 目录。

## 实现说明

`louvre.js` 是一段纯 Canvas 像素处理，把它搬到 Python 需要复刻不少浏览器细节，
这些都在 `oli_renderer.py` 里做了：

- **灰度**：`floor(r*.299 + g*.587 + b*.114)`
- **铅笔排线**：暗部二值掩膜按 1/4 缩放再放大，叠加纹理后乘 `shadeLight`
- **降噪 / 卷积**：均值卷积走积分图，其他核走通用卷积；边界按边缘复制
- **线稿**：`128 + 原图 − 模糊`，再按 `lightCut/darkCut` 做极端对比拉伸
- **上色**：沿对角线线性渐变取色，`alpha = 255 − 亮度`（原版就是这么"上色"的）
- **缩小再放大**（1.4 / 1.3）制造印刷柔化；缩放按**预乘 alpha** 并复刻 Canvas 的
  mipmap/双线性采样
- **水印**：浮点目标矩形 + 亚像素落点，预乘 alpha 合成
- **数值细节**：`Uint8ClampedArray` 的截断与四舍五入、JS `Math.round` 的 .5 进位

校验方式是拿本机 Edge（Playwright）跑原版 `louvre.js`、直接取 canvas 原始像素做真值，
逐像素比对了 18 组参数组合（含 3000px 大图与竖图），默认参数下平均误差约 0.4/255。

## 已知差异

- JPEG 解码器与浏览器不同，极端对比附近会有零星像素翻转（视觉无感）
- `zoom` 缩小 0.5 倍、`浮雕` 卷积这类"高对比放大"参数，误差略大（约 1/255）
- 原版的「叠加歌词」功能未完成，本插件也未实现

## 致谢与许可

- 原始项目：[itorr/one-last-image](https://github.com/itorr/one-last-image)，作者 [itorr](https://github.com/itorr)，MIT
- 素材 `pencil-texture.jpg`、`one-last-image-logo2.png` 来自原项目
- AstrBot 插件移植：[Unknownloli](https://github.com/Unknownloli)
- 本项目同样以 MIT 协议发布，见 [LICENSE](LICENSE)
