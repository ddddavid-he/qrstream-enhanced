# QRStream

[English](README.md) · [网页接收器](https://ddddavid-he.github.io/qrstream-enhanced/) · [完整使用指南](docs/USAGE-zh.md)

通过二维码流传输文件：电脑显示二维码，手机摄像头接收，或从录制的视频中恢复文件。默认使用 RaptorQ；命令行也兼容旧版 LT 码流。

## 快速开始

需要 Python 3.10 或更新版本。通过 pip 安装，也可使用 `uv tool install qrstream`：

```bash
pip install qrstream
```

在电脑上显示文件对应的二维码：

```bash
qrstream encode report.pdf
```

手机打开 **[网页接收器](https://ddddavid-he.github.io/qrstream-enhanced/)**，允许使用摄像头，对准电脑屏幕，点击红色快门开始接收。恢复完成后保存文件。

网页端支持 **V4 / RaptorQ** 码流，画面和文件均在设备上处理，不上传。下载文件名为 `qrstream-output.bin`，保存后改为原文件扩展名即可。

## 相机操作

- **红色快门**：开始或暂停接收；暂停保留已收集的数据块。
- **停止**：清空数据块和进度，保持摄像头预览；已有数据时先确认。
- **分辨率**：点击左上角文字切换设备支持的档位，保留文件进度。
- **详情与保存**：查看接收状态，或保存恢复后的文件；进度环在恢复完成后才到 100%。

## 视频文件

生成二维码视频，或从录制的视频恢复文件：

```bash
qrstream encode report.pdf -o report.mp4
qrstream decode recording.mp4 -o recovered.pdf
```

`qrs` 是短命令别名。查看参数可运行 `qrstream encode --help` 或 `qrstream decode --help`。

## 文档与开发

- [完整使用指南](docs/USAGE-zh.md)：全部参数、校准、Python API、网页开发和相关说明。
- [架构文档](docs/ARCH.md)：模块、协议与测试工具。
- [贡献指南](docs/CONTRIBUTING.md)：开发、分支、CI 和发布流程。

从仓库安装开发环境：

```bash
git clone https://github.com/ddddavid-he/qrstream-enhanced.git
cd qrstream-enhanced
uv sync --dev
```

网页改动合入 `main` 后，由 [Pages 工作流](.github/workflows/pages.yml) 构建部署。

## 许可证

MIT
