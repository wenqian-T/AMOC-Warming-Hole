# 数据来源与处理层级

本次按最终代码的文件读取路径选取数据，未重新下载或重新计算原始数据。

|目录|来源及用途|
|---|---|
|data/historical_analysis/annual|本项目既有 CMIP6 historical 年均预处理数据；12 模式各 5 成员|
|data/CMIP6|上述模式的 historical areacello 海洋网格面积文件；保留代码搜索时可能使用的成员候选|
|data/OBS/ERSSTv5|ERSSTv5 月均 SST 观测|
|data/OBS/HadISST|HadISST SST 观测|
|data/OBS/RAPID|RAPID transports v2024-2 观测|
|data/ssp585_analysis/annual/maps_2deg.nc|本项目既有 SSP585 两度网格预计算地图|
|data/ssp585_analysis/tables|本项目既有未来分析指标及 1950–2100 年序列|

CMIP6 原数据由各模式机构通过 ESGF 分发；观测来源分别为 NOAA ERSST、Met Office HadISST 和 RAPID 项目。使用及再分发应遵循提供方条件并保留引用。本包未替数据提供者声明统一开源许可，也未核验每个数据集的完整授权条款。

原始路径均位于 project2/new 下。DATA_MANIFEST.csv 保留其相对结构，可用于追溯本地来源。

热输送量级参数代码引用 Wang et al. (2023)，DOI 10.1029/2023EA002887；参数化结果不代表 SPNA 区域闭合热收支或因果归因。
