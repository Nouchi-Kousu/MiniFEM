# MiniFEM —— 基于 Python 的轻量二维有限元分析工具

> 一个用于二维平面应力问题的小型有限元分析程序，包含自动三角网格生成、常应变三角单元求解器以及基本的后处理可视化模块。

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![UV](https://img.shields.io/badge/UV-0.6.4-d56ae1)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/License-MIT-9e2013)](https://github.com/Nouchi-Kousu/MiniFEM/blob/main/LICENSE)

## 快速开始

安装依赖项

```bash
uv sync
```

开始

```bash
uv run main.py
```

## 主要功能

网格自动划分：
![网格自动划分](./img/seed.gif)

计算结果及载荷绘制：
![计算结果及载荷绘制](./img/load23.png)