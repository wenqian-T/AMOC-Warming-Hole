# 运行数据

GitHub 代码包不包含大体积运行输入；完整交付包的同一目录包含全部输入。
从完整包复制整个 data 目录到仓库根目录，保持子目录结构不变，再运行：

```bash
python scripts/check_inputs.py
python scripts/check_inputs.py --sha256
```

文件列表及 SHA256 见根目录 DATA_MANIFEST.csv。尚未设置外部数据下载链接。
