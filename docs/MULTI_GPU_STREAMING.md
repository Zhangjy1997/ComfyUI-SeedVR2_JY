# 多 GPU 分段超分

原来的多卡 CLI 会在 worker 中积累全部输出，再把各卡的结果拼成完整张量。
即使设置 `--chunk_size`，输出端的内存也会随视频时长增长。

现在多卡视频路径改为：规划帧范围 → GPU 领取片段 → 小块推理并立即编码 →
保存片段文件 → 领取下一段 → FFmpeg 按顺序合并。进程间只传任务信息和完成状态。
不需要事先把源视频切成多个文件，也不会把所有超分帧传回主进程。

## 使用示例

需要 FFmpeg 在 PATH 中。以下以两张 GPU、输出短边 2160 像素为例；
对于 16:9 横屏视频，输出为 3840×2160。模型、VAE tiling 和卸载参数可沿用现有配置。

```bash
python inference_cli.py input.mp4 \
    --output output_4k.mp4 \
    --cuda_device 0,1 \
    --resolution 2160 \
    --segment_duration 60 \
    --segment_overlap 4 \
    --chunk_size 81 \
    --batch_size 33 \
    --temporal_overlap 3 \
    --cache_dit --cache_vae \
    --video_backend ffmpeg
```

这是一组说明参数关系的起始配置，并非已在特定 GPU 上验证的性能配置。
实际显存不足时需要调整 batch、VAE tiling 和模型卸载设置。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--segment_duration` | 每个 GPU 任务负责的视频时长，单位秒 | 60 |
| `--segment_overlap` | 任务片段前后额外读取的上下文帧数，编码前裁掉 | 4 |
| `--chunk_size` | 每次推理新读入的帧数，处理后立即写入片段视频 | 多卡为 `max(33, batch_size)`；显式设置优先 |
| `--batch_size` | 模型内部一次处理的帧数 | 5 |
| `--temporal_overlap` | 同一推理块内批次的融合，以及小块之间的输入上下文 | 0 |
| `--cache_dit --cache_vae` | 在同一 worker 中跨小块和片段复用模型 | 关闭 |

`--chunk_size 0` 在单卡模式仍表示整段读取，多卡模式则自动采用上述有限大小。
上下文帧和首段 `--prepend_frames` 会使实际推理帧数略大于 chunk_size。
模型缓存开启时，未指定卸载设备会沿用 CLI 的 CPU 默认卸载行为。

## 内存和并发

一分钟只是任务和文件的边界，不会先在内存中攒满一分钟再编码。
内存主要取决于同时工作的 GPU 数、chunk_size、分辨率和模型设置。
调度队列中没有图像张量，输出格式转换也只针对单帧。

每张 GPU 同时执行一个任务，完成后领取下一个。任务少于 GPU 时只启用需要的 worker；
不足一分钟的视频默认只有一个任务。测试短视频多卡速度时，可以调小 segment_duration。

## overlap、帧数和音频

每个任务有唯一的输出帧范围，前后 overlap 只作为额外输入上下文。
编码时只保留该范围，所以不会因 overlap 或 prepend 增加最终帧数。
本版没有对相邻任务的输出做交叉融合；裁掉重复帧不保证消除画质接缝，需实测确认。

最终视频通过 FFmpeg 复制已编码的视频流合并，不会再对整段视频解码、重新超分或重新编码视频。
源文件有音频时，会按 skip_first_frames / load_cap 对应的时间范围提取，并编码成 AAC。
延续原 CLI 的 OpenCV 固定帧率处理方式，不保留可变帧率的原始时间戳。
PNG 输出按全局帧序号保存，不包含音频。

## 临时文件和失败处理

片段存放在输出目录旁的 `.seedvr2-segments-*` 临时目录中。
成功合并后自动清理；失败时保留片段、`segments.json` 和 `FAILED.txt`，错误会显示路径。
已有目标视频仅在合并成功后才被替换。当前没有自动断点续跑功能。

磁盘需要容纳全部编码片段和最终视频，空间消耗约为两份编码输出加少量日志，
不会保存全量浮点图像。worker 抛错或意外退出时，调度器会停止其他 worker 并报告失败。

## 开发验证

```bash
python -m unittest discover -s tests -v
```

调度测试不依赖 GPU；视频测试使用 PyTorch、NumPy、OpenCV 和 FFmpeg，
将模型推理替换为恒等变换，验证真实读写、帧顺序、裁剪和片段合并。
真实 SeedVR2 推理的显存峰值、速度及边界画质仍需在目标 GPU 上验证。
