# Color Recognition and Correction System

本项目用于胶块颜色校正与颜色识别。目标是在手机拍摄、光照变化、背景反射和样品高光/阴影存在的情况下，对待测胶块进行颜色校正，并在 128 个标准颜色中输出最接近的颜色编号、名称、TopK 结果与置信度。

项目目前采用的主流程是：

```text
输入图片
↓
ColorChecker 色卡标定
↓
root polynomial regression 色彩校正
↓
ROI 稳健取色与局部背景补偿
↓
Lab residual 回归模型修正
↓
与 128 个标准 Lab 计算 ΔE2000
↓
输出 W001-W128 识别结果、TopK 与置信度
```

## 当前功能

- 支持手机实拍图片的颜色校正实验（数据采用iphone17）
- 支持 ColorChecker 四角手动标定
- 支持 root polynomial regression / ridge 参数实验
- 支持 128 色标准 Lab 数据表
- 支持整版 128 色胶块数据采集与评估
- 支持局部背景 Lab 补偿
- 支持 single residual model 修正 Lab 残差
- 支持单个未知胶块识别
- 输出 Top1 / Top2 / Top3 / TopK 识别结果
- 输出 ΔE2000、置信度、overlay 可视化结果
- 支持多张 run 结果汇总与 leave-one-run-out 验证

## 项目状态

本项目目前处于实验验证与工程化整理阶段。

已经完成：

```text
传统色彩校正 baseline
root_poly2 / ridge 参数实验
128 色标准数据引入
轻微背景补偿
多张常态光数据采集
Lab residual 回归模型训练
单个胶块识别 demo
```

当前推荐主线：

```text
real_scene_rootpoly_pipeline_v2.py
train_single_residual_model.py
single_predict_v2.py
```

历史实验脚本和输出目录保留用于对比，但不是最终推荐入口。

## 主要文件说明

```text
data.csv
```

128 个标准颜色的 Lab 数据表。

格式示例：

```csv
W001,大红,"44.5, 46.99, 19.2"
W002,正红,"39.73, 39.58, 14.35"
W003,中国红,"43.52, 53.88, 20.92"
```

```text
standard_chart.png
```

标准 ColorChecker 图片，用于提取 24 色标准 RGB。

```text
real_scene_rootpoly_pipeline_v2.py
```

整版 128 色样本处理脚本。用于从实拍整版图片中提取 128 个 ROI，进行 rootpoly2 校正、背景补偿、TopK 识别和结果统计。

主要输出：

```text
best_target_results.csv
candidate_summary.csv
group_summary_best.csv
report.json
selected_rois.json
chart_corners.json
```

```text
summarize_runs.py
```

用于汇总多个 `dataset_runs/run_xxx` 的结果，生成整体统计、每个色号统计和每个色系分组统计。

主要输出：

```text
summary_out/runs_summary.csv
summary_out/code_summary.csv
summary_out/group_summary_all_runs.csv
summary_out/summary_report.txt
```

```text
train_single_residual_model.py
```

训练适用于单个未知胶块识别的 Lab residual 回归模型。

该模型不使用：

```text
idx
row_norm
col_norm
code
```

因此不会依赖 128 色板的固定位置，适合迁移到单个未知胶块识别场景。

模型输入包括：

```text
raw Lab
rootpoly2 校正后 Lab
背景补偿后 Lab
局部背景 Lab
chroma 色度
hue 色相
颜色族特征
```

模型输出：

```text
ΔL, Δa, Δb
```

即对当前测量 Lab 的残差补偿。

```text
single_predict_v2.py
```

单个未知胶块识别脚本。

功能包括：

```text
ColorChecker 标定
单个 ROI 框选
ROI 高光/阴影剔除
局部背景提取
bg0.25 背景补偿
single residual model 修正
128 标准色 ΔE2000 最近邻识别
TopK 输出与置信度判断
```

## 安装依赖

建议使用 Python 3.10 或以上版本。

```bash
pip install numpy opencv-python scikit-image
```

如果后续训练分类模型，可额外安装：

```bash
pip install scikit-learn
```

如果使用 XGBoost：

```bash
pip install xgboost
```

## 整版 128 色样本处理

示例命令：

```powershell
python real_scene_rootpoly_pipeline_v2.py `
  --photo pic_all.jpg `
  --standard-chart standard_chart.png `
  --standards-csv data.csv `
  --target-codes auto:1-128 `
  --eval-count 128 `
  --out dataset_runs/run_001 `
  --background-lab "84.71,-1.14,-3.64" `
  --bg-strength-list 0.25 `
  --residual-strength-list 0 `
  --bg-bright-percentile 70 `
  --bg-max-chroma 18
```

每张新图片建议单独输出到一个 run 目录：

```text
dataset_runs/run_001
dataset_runs/run_002
dataset_runs/run_003
...
```

## 汇总多个 run

```powershell
python summarize_runs.py `
  --runs-glob "dataset_runs/run_*" `
  --eval-count 128 `
  --out summary_out
```

重点查看：

```text
summary_out/runs_summary.csv
summary_out/code_summary.csv
summary_out/group_summary_all_runs.csv
summary_out/summary_report.txt
```

## 训练 single residual 模型

clean 版本示例：

```powershell
python train_single_residual_model.py `
  --runs "dataset_runs/run_001,dataset_runs/run_004,dataset_runs/run_006,dataset_runs/run_007,dataset_runs/run_008,dataset_runs/run_009,dataset_runs/run_010,dataset_runs/run_012,dataset_runs/run_013,dataset_runs/run_014" `
  --standards-csv data.csv `
  --baseline-file best_target_results.csv `
  --eval-count 128 `
  --background-lab "84.71,-1.14,-3.64" `
  --out single_residual_model_out_clean
```

输出：

```text
single_residual_model_out_clean/alpha_sweep_summary.csv
single_residual_model_out_clean/loo_predictions_best_alpha.csv
single_residual_model_out_clean/train_predictions_full_model.csv
single_residual_model_out_clean/single_residual_model.json
```

其中 `single_residual_model.json` 是单个胶块识别时使用的模型文件。

## 单个未知胶块识别

推荐使用 v2 版本：

```powershell
python single_predict_v2.py `
  --photo single_test.jpg `
  --standard-chart standard_chart.png `
  --standards-csv data.csv `
  --model single_residual_model_out_clean/single_residual_model.json `
  --out single_predict_out
```

运行时需要：

```text
1. 点选 ColorChecker 四角
2. 框选待测胶块 ROI
```

输出：

```text
single_predict_result.json
single_predict_topk.csv
single_predict_overlay.jpg
target_crop_original.jpg
target_crop_root_corrected.jpg
chart_corners.json
target_roi.json
```

如果是黄色、浅黄、金色等容易受高光影响的色块，可以尝试：

```powershell
python single_predict_v2.py `
  --photo single_test_yellow.jpg `
  --standard-chart standard_chart.png `
  --standards-csv data.csv `
  --model single_residual_model_out_clean/single_residual_model.json `
  --out single_predict_yellow `
  --use-family-filter `
  --roi-center-ratio 0.65 `
  --roi-trim-percent 12
```

## 识别结果说明

单个识别结果会输出：

```text
Top1 编号与名称
Top2 编号与名称
Top3 编号与名称
ΔE2000
Top2 margin
confidence
```

置信度规则：

```text
high      Top1 ΔE 较低，且 Top1 与 Top2 差距明显
medium    Top1 较可信，但存在相近颜色
low       Top1 可能正确，但需要结合 Top2/Top3 判断
very_low  不建议自动判定，需要人工复核
```

对于红色、黄色、浅色、灰色等相近颜色，不建议只看 Top1，应结合 Top3 和置信度判断。

## 当前实验结果

在多张 128 色实拍样本上，使用 leave-one-run-out 验证 single residual model：

```text
before mean ΔE：约 15.93
rootpoly2 + bg0.25 后 mean ΔE：约 9.90
single residual model 后 mean ΔE：约 5.78
model max ΔE：约 17.39
model p95 ΔE：约 11.92
```

说明 residual 模型能够有效补偿 ColorChecker 全局校正后，在胶块区域仍然存在的系统误差。



## 拍摄建议

为了提高识别稳定性，建议：

```text
固定光源
固定相机焦距
尽量锁曝光和白平衡
不要使用滤镜、夜景、人像、美化
尽量保留原图，不要微信压缩
图片中包含 ColorChecker
背景材料尽量一致
ROI 框选时避开明显高光和暗阴影
```

黄色、浅色和灰色对光照最敏感，需要特别注意高光和阴影。

## 目录说明

当前仓库中保留了大量实验输出目录，例如：

```text
output_rootpoly_stable
output_known_bg_v2
output_final_bg025
output_pure_rootpoly2_all128
single_predict_out
single_residual_model_out_clean
summary_out
```

此目录主要用于实验记录和效果对比。