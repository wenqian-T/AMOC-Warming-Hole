# 交付整理变更

- 将最终代码中的本机 project2/new 路径改为 PROJECT_ROOT/data，本机 results_final 路径改为 PROJECT_ROOT/results。
- 添加通过 PROJECT_ROOT.marker 定位项目根目录的单元；支持 notebook 和脚本从项目内运行。
- 清空 notebook 的旧输出和执行序号；原结果另存 reference_results。
- 保留原代码单元的顺序、公式、参数及重复绘图单元。
- fig6 原文件仅有绘图单元。新增统计准备单元，从 Fig02_SST_member_trends.csv 的 Global-SPNA 与 AMOC_member_trends.csv 的 1950-2014 成员趋势配对，计算模式均值、样本标准差、OLS、Pearson 相关及留一法。
- 新增依赖、数据清单、顺序运行入口和输入校验工具。
- 原始最终代码与源数据没有改动。
