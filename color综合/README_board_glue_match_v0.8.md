下面这版可以直接保存成：

```text
README_board_glue_match_v0.8.md
```

---

# 板材-胶块颜色匹配方案说明 v0.8

## 1. 最终目标

本项目最终目标不是单纯追求校正后 Lab 与机采标准 Lab 的 ΔE 最低，而是实现：

```text
用户肉眼看到某块板材和某几个胶块颜色接近
↓
系统自动计算后，也能推荐出这些相近胶块
↓
输出图像中，板材和推荐胶块的显示效果也接近人眼观感
```

因此当前评价重点从：

```text
校正后胶块 Lab vs 机采标准 Lab
```

转为：

```text
校正/视觉处理后的板材
vs
128 胶块视觉库
```

最终核心指标是：

```text
机器 Top1 / Top3 / Top5 是否覆盖人眼认为相近的胶块
```

ΔE 仍然保留，但主要作为排序和参考指标，而不是唯一真理。

---

## 2. 当前整体流程

当前流程分为两端：

```text
A. 胶块端：建立 128 胶块视觉库
B. 板材端：拍摄板材 + 色卡，并匹配胶块库
```

整体结构为：

```text
128 胶块图
↓
ColorChecker 基础校正
↓
胶块 residual 视觉修正
↓
生成胶块视觉库

板材 + 色卡照片
↓
ColorChecker 基础校正
↓
板材视觉显示修正 / 匹配空间转换
↓
与胶块视觉库计算 ΔE
↓
输出 TopK 推荐胶块
```

---

## 3. 胶块端方案

胶块端当前最优版本为：

```text
corrected residual + manual circle + original background + series visual rules v0.7
```

它不是单纯的整图校色，而是两条链路：

```text
A 链路：ColorChecker 校正
    用于获得较可靠的胶块基础颜色。

B 链路：视觉 residual 渲染
    用于让最终胶块图更接近肉眼观感。
```

胶块端最终图的原则：

```text
背景尽量保留原图现场感
胶块区域使用 ColorChecker 校正后的结果
胶块颜色再按标准 Lab residual 进行适度补偿
对容易过修的色系加入分组视觉保护规则
```

---

## 4. 胶块端输入文件

胶块端需要：

```text
pic_all.jpg
    128 色胶块原始拍摄图。

standard_chart.png
    标准 ColorChecker 色卡图。

data.csv
    128 胶块机采 Lab 标准数据。

output_128/report.json
    main.py 输出的主报告文件。

output_128/02_corrected.png
    ColorChecker 校正后的整图。

output_128/corrected_residual_alpha_sweep/visual_circles_manual.json
    手动画圆得到的 128 个胶块视觉区域。
```

其中：

```text
report.json
    用于读取胶块编号、标准 Lab、校正后 Lab、ROI、ΔE 等信息。

visual_circles_manual.json
    只控制最终图中哪里被视觉修正，不改变测色 ROI。
```

---

## 5. 胶块端运行方式

### 5.1 先运行 ColorChecker 基础校正

```powershell
python main.py `
  --photo pic_all.jpg `
  --standard standard_chart.png `
  --data data.csv `
  --out output_128 `
  --top-k 10
```

该步骤输出：

```text
output_128/report.json
output_128/02_corrected.png
```

当前基础校正结果约为：

```text
Target mean ΔE: 9.625 -> 5.564
```

说明 ColorChecker 校正对基础测色有效。

---

### 5.2 确认手动画圆文件存在

```powershell
Test-Path output_128\corrected_residual_alpha_sweep\visual_circles_manual.json
```

返回 `True` 即可复用。

该文件非常重要，建议备份。

---

### 5.3 运行当前胶块视觉最优版

```powershell
python visual_alpha_sweep_eval_series_rules.py `
  --report output_128/report.json `
  --photo pic_all.jpg `
  --alpha-list "1.0" `
  --background-mode original `
  --bg-scale 0 `
  --use-visual-series-rules `
  --rule-strength 0.7
```

如果程序没有自动找到画圆文件，则手动指定：

```powershell
python visual_alpha_sweep_eval_series_rules.py `
  --report output_128/report.json `
  --photo pic_all.jpg `
  --alpha-list "1.0" `
  --background-mode original `
  --bg-scale 0 `
  --use-visual-series-rules `
  --rule-strength 0.7 `
  --visual-circle-file output_128/corrected_residual_alpha_sweep/visual_circles_manual.json
```

---

## 6. 胶块端当前最佳参数

```text
alpha = 1.0
background_mode = original
bg_scale = 0
use_visual_series_rules = True
rule_strength = 0.7
```

含义如下：

```text
alpha = 1.0
    胶块区域完整叠加 corrected_lab 到 standard_lab 的 residual 方向。

background_mode = original
    最终背景完全使用原图，保留现场感。

bg_scale = 0
    不额外中性化背景。

use_visual_series_rules = True
    开启 128 色分组视觉保护规则。

rule_strength = 0.7
    分组保护规则生效 70%，保留 30% 原始 residual 效果。
```

当前该参数组合下，虽然视觉图相对机采标准值的 ΔE 可能略升，但肉眼效果更自然、更接近现场观察。

---

## 7. 胶块端输出文件

主要输出目录：

```text
output_128/corrected_residual_alpha_sweep/
```

关键文件：

```text
preview_alpha_1.00.png
    当前胶块视觉预览图。

alpha_sweep_summary.csv
    alpha 对应的 ΔE 统计。

alpha_sweep_result.json
    本次运行参数与输出信息。

visual_rules_alpha_1.00.csv
    每个色号实际使用的视觉规则记录。

visual_circles_manual.json
    手动画圆结果，必须备份。
```

建议将当前最佳图另存为：

```text
preview_best_rule07.png
```

---

## 8. 胶块视觉库

胶块视觉库是后续板材匹配的基础。

库文件建议为：

```text
output_128/glue_visual_library/glue_visual_library.csv
```

建议字段包括：

```text
code
name
machine_L
machine_a
machine_b
corrected_L
corrected_a
corrected_b
visual_L
visual_a
visual_b
visual_crop_path
rule_info
```

其中：

```text
machine_Lab
    机采标准值，只作为参考。

corrected_Lab
    ColorChecker 基础校正后的胶块 Lab。

visual_Lab
    当前视觉流程下胶块最终呈现出的 Lab，用于板材匹配。

visual_crop_path
    胶块视觉展示图路径。
```

后续正式匹配主要使用：

```text
visual_L, visual_a, visual_b
```

---

## 9. 板材端方案

板材端目标是：

```text
用户单独拍摄 板材 + 色卡
↓
系统校正后得到肉眼较接近的板材图
↓
系统在 128 胶块视觉库中匹配出相近胶块
```

板材端不能使用：

```text
standard_lab - corrected_lab
```

因为未知板材没有标准 Lab。

因此板材端使用的是：

```text
基础色卡校正 + 当前视觉匹配空间/显示修正参数
```

最终匹配逻辑为：

```text
board_visual_Lab
vs
glue_visual_Lab
```

计算：

```text
E_match = ΔE(board_visual_Lab, glue_visual_Lab)
```

并输出 TopK。

---

## 10. 板材端当前运行方式

当前成功参数示例：

```powershell
python board_photo_match_v5.py `
  --photo board3.jpg `
  --standard standard_chart.png `
  --library output_128/glue_visual_library/glue_visual_library.csv `
  --mapping output_128/glue_visual_library/visual_mapping_T_poly2/visual_mapping_T.json `
  --out board_match_try1 `
  --top-k 10 `
  --board-display-l-offset 5.0 `
  --board-display-chroma-scale 0.95 `
  --board-display-chroma-offset 0 `
  --board-display-a-offset 0.3 `
  --board-display-b-offset 2.5
```

参数说明：

```text
--photo
    输入板材 + 色卡照片。

--standard
    标准 ColorChecker 色卡图。

--library
    128 胶块视觉库。

--mapping
    corrected Lab 到 visual Lab 的视觉映射模型。

--out
    输出目录。

--top-k
    输出前 K 个候选胶块。

--board-display-l-offset
    板材显示亮度微调。

--board-display-chroma-scale
    板材显示色度缩放。

--board-display-chroma-offset
    板材显示色度偏移。

--board-display-a-offset
    a 通道显示偏移。

--board-display-b-offset
    b 通道显示偏移。
```

---

## 11. 板材端输出建议

每个板材案例建议保存：

```text
board_original.jpg
    原始板材 + 色卡照片。

board_corrected.png
    基础校正后的板材图。

board_display.png
    最终显示效果图。

board_roi_preview.png
    板材取色 ROI 预览图。

top10.csv
    Top10 胶块匹配结果。

match_preview.png
    板材与 TopK 胶块的对比预览图。
```

Top10 表建议包含：

```text
rank
code
name
E_match
glue_visual_L
glue_visual_a
glue_visual_b
board_visual_L
board_visual_a
board_visual_b
visual_crop_path
```

---

## 12. 当前评价方式

当前不再以单一机采标准 ΔE 作为最终评价。

评价分为三层：

```text
1. 板材显示效果
    板材校正/显示图是否接近肉眼看到的板材。

2. 胶块显示效果
    胶块视觉库图是否接近真实胶块。

3. 匹配效果
    机器 Top1 / Top3 / Top5 是否包含人眼认为相近的胶块。
```

建议记录表：

```text
case_id
board_image
manual_best
algorithm_top1
manual_best_in_top3
manual_best_in_top5
top1_acceptable
note
```

判断标准：

```text
Top1 可接受：
    系统推荐第一名肉眼看也接近。

Top3 覆盖：
    人眼最像的胶块在算法前三名内。

Top5 覆盖：
    人眼最像的胶块在算法前五名内。
```

---

## 13. 当前已完成状态

当前已完成：

```text
1. 128 胶块单张图 ColorChecker 校正。
2. 胶块基础校正 mean ΔE 约为 5 左右。
3. 胶块 v0.7 视觉 residual 方案，肉眼效果较好。
4. 单拍板材 + 色卡流程已经跑通。
5. 板材输出图肉眼效果不错。
6. 板材可以匹配到肉眼也还可以的胶块。
```

这说明当前已经完成最小闭环：

```text
板材拍摄
↓
板材校正/显示
↓
胶块库匹配
↓
TopK 肉眼验证
```

---

## 14. 后续工作

后续重点不再是继续大改算法，而是验证稳定性。

建议下一步：

```text
1. 固化当前版本
    保存所有脚本、参数、输入图、输出图和库文件。

2. 拍摄 5 块不同颜色板材
    每块拍 2~3 张，均带色卡。

3. 每张图输出 Top10
    保存匹配结果和对比图。

4. 人工记录 Top1 / Top3 / Top5 是否合理
    形成小规模验证表。

5. 根据失败案例分析问题
    区分是取色 ROI 问题、显示修正问题，还是胶块库本身问题。
```

当前阶段的主线应从：

```text
继续调效果
```

转为：

```text
固定版本 + 多样本验证 + TopK 覆盖率统计
```

---
 