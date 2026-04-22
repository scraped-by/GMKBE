# baselines

该目录收录论文实验中使用的两个 baseline 实现，并保留各自独立的源码结构。

- `SGMD22-MBPE`：对应论文 `Efficient Algorithms for Maximal k-Biplex Enumeration`，在本文实验中对应的 baseline 名称为 `iTraversal`。
- `MaximalBiPlex`：对应论文 `Efficient Maximal Biplex Enumerations with Improved Worst-Case Time Guarantee`，在本文实验中对应的 baseline 名称为 `BPPivot`。

为便于统一实验流程，目录中的部分工程文件与数据读取接口做了兼容性适配，使其能够在当前仓库的构建方式和预处理后数据格式下运行。

如需具体的编译或运行方式，请分别参考各子目录中的构建文件与说明文档。
