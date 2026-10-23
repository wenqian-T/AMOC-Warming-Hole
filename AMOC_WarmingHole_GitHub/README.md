# 北大西洋增暖空洞与 AMOC

分析 CMIP6 historical 与 SSP5-8.5 的海表温度、AMOC 和海表热通量，并与 ERSSTv5、HadISST 和 RAPID 观测比较。

## 完整数据下载

[我用夸克网盘给你分享了「AMOC_WarmingHole_Complete.zip」，点击链接或复制整段内容，打开「夸克APP」即可获取。
/~df613bgXhC~:/
链接：https://pan.quark.cn/s/df80f1fecc7d]
解压后，将其中的 data 文件夹放在本仓库根目录，
再按照上面的运行步骤执行。
## 内容

- `notebooks/`：6 个最终 notebook 的可移植副本，已清除旧输出。
- `scripts/`：逐 notebook 导出的 Python 脚本及顺序运行入口。
- `data/`：完整交付包中的运行输入；GitHub 代码包只含说明。
- `reference_results/`：原项目已有结果，用于对照；并非本次重新运行生成。
- `results/`：重新运行生成的文件。
- `docs/`：已有的一页预估表与技术报告附录 PDF。
- `DATA_MANIFEST.csv`：数据文件相对位置、大小和 SHA256。
- `VALIDATION.md`：本次验证范围与结果。

## 安装与运行

建议 Python 3.11 或 3.12，在项目根目录运行：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_inputs.py
MPLBACKEND=Agg python scripts/run_all.py
```

Windows 可先设置环境变量 MPLBACKEND=Agg，再执行 `python scripts/run_all.py`。
也可运行 `jupyter lab`，按以下顺序依次打开 notebook，并使用 Restart Kernel and Run All：

1. `fig1_2.ipynb`：历史 SST 地图与指数，输出 Fig01、Fig02。
2. `fig3_4_5.ipynb`：历史 AMOC 与 RAPID，输出 Fig05、Fig06、Fig07 等。
3. `fig6.ipynb`：历史 AMOC–WHI 关系及留一法，输出 Fig08。
4. `fig7_8_9.ipynb`：历史 hfds 和热输送量级，输出 Fig11–13。
5. `fig10_11.ipynb`：未来 SST，输出 Fig14–16。
6. `fig12_13.ipynb`：未来 AMOC 与 WHI，输出 Fig20、Fig21。

文件名与图内编号沿用原代码，两者并不一一对应。原 notebook 的重复绘图单元保留，后面的单元可能覆盖前面保存的同名图片。

`fig7_8_9` 需要 `fig3_4_5` 生成的 AMOC_member_annual.csv；`fig6` 需要前两个 notebook 的成员趋势结果。请遵守运行顺序。
Cartopy 首次绘图可能联网下载 Natural Earth 地图底图；其缓存不是气候输入数据的一部分。

## 数据与可复现范围

完整包带有代码直接读取的历史年均 NetCDF、网格面积、观测文件，以及 SSP585 预计算地图和指标表。没有打包原始月度 CMIP6 全数据库，也没有声称能从原始月度文件重新生成全部年均预处理结果。

历史计算使用 12 模式、每模式 5 成员；未来分析读取预计算的 5 模式、每模式 5 成员数据。WHI 在代码中使用全球海洋 SST 减去 SPNA SST。

数据来源与使用条件见 `DATA_SOURCES.md`。数据清单记录了本地文件的校验值，不等于对数据生产者公布校验值的认证。

## GitHub 上传

本代码包可作为仓库根目录。不要把完整 ZIP 直接提交到仓库；`data/` 中的大文件已被 .gitignore 排除。
完整数据包应单独提供下载渠道，发布后在本 README 补上实际下载链接；当前尚未创建远程仓库或上传数据。

仓库应包含代码、README、依赖、数据来源和获取说明、数据清单、验证记录，以及课程要求的报告与必要结果。未找到课程评分要求原文，因此不能保证这些文件覆盖课程全部必交项。
代码许可证须由作者决定；未擅自添加开源许可证。数据的使用权限由各数据提供方决定。
