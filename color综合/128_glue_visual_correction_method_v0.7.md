# 128 色胶块视觉校正方案说明（当前最优版 v0.7）

## 1. 方案定位

本方案用于在已知 128 色胶块拍摄场景下，生成一张更接近肉眼观感的结果图。

当前版本不是单纯追求整张图的统一色彩校正，而是将流程拆成两条链路：

```text
A 链路：ColorChecker 色彩校正，用于获得较可靠的胶块颜色基础
B 链路：视觉渲染修正，用于让最终图更接近人眼看到的效果
```

最终输出图的原则是：

```text
背景尽量保留原图现场感
胶块区域使用 ColorChecker 校正后的结果
胶块颜色再按标准 Lab residual 进行适度补偿
对容易过修的色系加入分组视觉保护规则
```

当前人工筛选后的最佳参数为：

```powershell
python visual_alpha_sweep_eval_series_rules.py --report output_128/report.json --photo pic_all.jpg --alpha-list "1.0" --background-mode original --bg-scale 0 --use-visual-series-rules --rule-strength 0.7
```

---

## 2. 输入文件

运行当前版本需要以下文件：

```text
pic_all.jpg
    原始拍摄图像。

output_128/report.json
    main.py 生成的主报告文件。
    包含每个胶块的编号、标准 Lab、ROI、校正前后 Lab、ΔE、TopK 等信息。

output_128/02_corrected.png
    ColorChecker 校正后的整图。
    如果命令中不显式传入 --corrected，程序默认读取 report.json 同目录下的 02_corrected.png。

output_128/corrected_residual_alpha_sweep/visual_circles_manual.json
    手动画圆得到的 128 个胶块视觉区域。
    该文件只影响最终图中“哪里被上色”，不影响原本 report.json 的测色 ROI。
```

---

## 3. 整体处理流程

当前版本完整流程如下：

```text
1. 读取原图 pic_all.jpg
2. 读取 ColorChecker 校正图 output_128/02_corrected.png
3. 读取 report.json 中 128 个胶块的标准 Lab 与校正后测得 Lab
4. 读取手动画圆结果 visual_circles_manual.json
5. 对每个胶块计算 residual：
       residual = standard_lab - corrected_measured_lab
6. 在 corrected 图的圆形胶块区域内叠加 residual
7. 对部分容易过修的色系启用 128 色分组视觉规则
8. 胶块区域使用修正后的结果
9. 背景区域使用原图
10. 合成最终 preview_alpha_1.00.png
```

最终图可以理解为：

```text
final_image = original_background + corrected_residual_glue_blocks
```

也就是：

```text
背景：来自原图
胶块：来自 corrected 图 + residual 修正 + 分组保护规则
```

---

## 4. A 链路：ColorChecker 校正

A 链路由 `main.py` 完成。

它的作用是利用 ColorChecker 色卡建立相机颜色响应校正模型，将原图从相机拍摄颜色校正到更接近标准色的状态。

之前 128 胶块的整体结果为：

```text
Target mean ΔE: 9.625 -> 5.564
```

这说明 ColorChecker 校正对测色是有效的。

A 链路输出的关键文件是：

```text
output_128/02_corrected.png
output_128/report.json
```

其中：

```text
02_corrected.png
    作为 B 链路的胶块基础图。

report.json
    提供每个胶块的标准 Lab、校正后 Lab、ROI、ΔE 等信息。
```

---

## 5. B 链路：视觉 residual 渲染

B 链路不是重新做 ColorChecker 校正，而是在 A 链路结果基础上进行局部视觉修正。

对每个胶块：

```text
corrected_measured_lab = ColorChecker 校正图中该胶块测得的 Lab
standard_lab = data.csv 中该胶块的标准 Lab
residual = standard_lab - corrected_measured_lab
```

基础叠加公式为：

```text
preview_lab = corrected_lab + mask × alpha × residual
```

其中：

```text
mask
    手动画圆得到的胶块区域 mask，并带有 feather 羽化。

alpha
    residual 叠加强度。当前最佳为 1.0。

residual
    该胶块从 corrected 状态到标准 Lab 的差值方向。
```

注意：这一步只作用在手动画圆的胶块区域内，不作用于整个方框 ROI，也不作用于背景。

---

## 6. 手动画圆区域

之前使用方框 ROI 或自动椭圆时，部分胶块会出现：

```text
整个方框被染红 / 染蓝 / 染黄
```

因此当前版本改为手动画圆。

手动画圆的意义是：

```text
测色 ROI：
    仍然来自 report.json / main.py
    用于评估 ΔE，保持稳定

视觉圆形 ROI：
    来自 visual_circles_manual.json
    只用于控制最终图中哪里被视觉修正
```

这样可以避免把胶块外的背景也一起进行 residual 叠加。

手动画圆保存文件：

```text
output_128/corrected_residual_alpha_sweep/visual_circles_manual.json
```

该文件建议长期保留，不要删除。只要拍摄图尺寸和胶块位置不变，就可以复用。

---

## 7. 背景处理策略

当前最佳参数：

```text
--background-mode original
--bg-scale 0
```

含义如下：

```text
background-mode original
    最终图背景完全使用原图背景。

bg-scale 0
    不对背景做额外中性化处理。
```

选择这个策略的原因是：

```text
ColorChecker 校正后的整图背景容易发白、发淡；
但人眼现场看到的背景通常保留一定光照氛围；
因此最终图应保留原图背景，只替换胶块区域。
```

如果将来需要稍微清理背景，可以尝试：

```powershell
--background-mode blend --background-mix 0.10
```

含义是：

```text
背景 = 90% 原图 + 10% corrected 图
```

但当前最优版本不使用该模式。

---

## 8. 当前最佳参数说明

当前最佳命令：

```powershell
python visual_alpha_sweep_eval_series_rules.py --report output_128/report.json --photo pic_all.jpg --alpha-list "1.0" --background-mode original --bg-scale 0 --use-visual-series-rules --rule-strength 0.7
```

### 8.1 `--report output_128/report.json`

读取主流程报告。

该文件提供：

```text
128 色胶块编号
每个胶块标准 Lab
每个胶块校正后 Lab
原始测色 ROI
ΔE 统计
```

### 8.2 `--photo pic_all.jpg`

读取原图。

当前版本中，原图用于：

```text
1. 提供最终背景
2. 提供手动画圆坐标参考
3. 在需要时用于和 corrected 图进行局部合成
```

### 8.3 `--alpha-list "1.0"`

只输出 alpha=1.0 的结果。

这里 alpha 表示 residual 叠加强度：

```text
alpha = 0
    不叠加 residual，只使用 corrected 图中的胶块。

alpha = 0.5
    补一半 residual。

alpha = 1.0
    按当前 residual 完整补偿到标准方向。
```

经过人工比较，目前 alpha=1.0 的肉眼效果最好。

### 8.4 `--background-mode original`

最终背景完全使用原图背景。

这是为了解决：

```text
背景被 ColorChecker 校正后发白、发淡
```

### 8.5 `--bg-scale 0`

关闭背景中性化。

即不再对背景执行：

```text
a/b 向中性灰靠近
```

当前版本认为背景应尽量保留现场感，因此设置为 0。

### 8.6 `--use-visual-series-rules`

开启 128 色分组视觉规则。

原因是统一 alpha=1.0 虽然整体观感较好，但部分色系会出现过修：

```text
W033-W048：
    部分颜色被修得偏浅、偏黄

W049-W064：
    浅灰、灰白容易变暖、变白
    W053 尤其明显

W081-W095：
    深灰组被修得偏浅

W113-W127：
    棕咖、深色组可能被修得偏浅
```

所以需要按色系限制 residual 的作用。

### 8.7 `--rule-strength 0.7`

控制分组视觉规则的生效强度。

可以理解为：

```text
rule_strength = 0
    不启用保护规则，完全按原始 residual 修正。

rule_strength = 1.0
    分组保护规则完全生效。

rule_strength = 0.7
    分组保护规则生效 70%，保留 30% 原始 residual 效果。
```

当前人工判断 `0.7` 在“标准色靠近”和“肉眼自然”之间平衡较好。

---

## 9. 128 色分组视觉规则

当前版本不是所有颜色都吃满 residual，而是根据色号分组设置不同保护策略。

### 9.1 W017-W032

该组当前肉眼观察基本一致。

处理策略：

```text
基本保留 alpha=1.0 的修正效果
不做明显削弱
```

### 9.2 W097-W112

该组当前肉眼观察也基本一致。

处理策略：

```text
基本保留修正效果
只做轻微保护
```

### 9.3 W033-W048

该组容易被修得：

```text
更浅
更黄
```

处理策略：

```text
限制 b 方向 residual
减少变黄
让 L 更多接近原图，减少变浅
```

### 9.4 W049-W064

该组多为浅灰、灰白、银灰等颜色，对黄色变化非常敏感。

风险：

```text
很容易变暖白
很容易显得发黄、发白
W053 是典型异常点
```

处理策略：

```text
强限制 b 正向 residual
限制变黄
限制过度变亮
```

### 9.5 W053 单独保护

W053 是当前观察到的明显异常点。

它的 corrected Lab 和 standard Lab 差距较大，因此完整 residual 会把它推向：

```text
更亮
更黄
更暖白
```

处理策略：

```text
单独降低 b residual
单独限制 L 变化
防止变成异常暖白
```

### 9.6 W081-W095

该组为深灰、深色灰等。

肉眼观察问题：

```text
算法后偏浅
```

处理策略：

```text
L 通道更多回到原图
减少 corrected 图带来的提亮感
```

### 9.7 W113-W127

该组多为棕、咖、深色、土棕等。

肉眼观察问题：

```text
部分颜色偏浅
部分颜色 b 方向不应吃满
```

处理策略：

```text
L 通道部分回原图
b 通道适度限制
```

### 9.8 W096、W128

这两个颜色在当前肉眼观察中较接近实物。

处理策略：

```text
尽量保持当前修正效果
不做强保护
```

---

## 10. 当前输出文件

运行当前命令后，主要输出目录为：

```text
output_128/corrected_residual_alpha_sweep/
```

其中重要文件包括：

```text
preview_alpha_1.00.png
    当前最终预览图。

alpha_sweep_summary.csv
    alpha 对应的 ΔE 统计汇总。

alpha_sweep_result.json
    本次运行参数、输出路径、推荐信息。

visual_rules_alpha_1.00.csv
    每个色号实际使用的视觉规则记录。
    用于检查 W053、W033-W048、W049-W064 等是否被正确保护。

visual_circles_manual.json
    手动画圆结果。
    非常重要，建议备份。
```

建议将当前最佳图另存为：

```text
preview_best_rule07.png
```

避免后续重新运行时被覆盖。

---

## 11. 复用步骤

### 11.1 先运行 A 链路

```powershell
python main.py --photo pic_all.jpg --standard standard_chart.png --data data.csv --out output_128 --top-k 10
```

如果已经跑过，且 `output_128/report.json`、`output_128/02_corrected.png` 存在，可以跳过。

### 11.2 确认手动画圆文件存在

```powershell
Test-Path output_128\corrected_residual_alpha_sweep\visual_circles_manual.json
```

如果返回 `True`，说明可以复用之前画的圆。

如果返回 `False`，需要重新手动画圆。

### 11.3 运行当前最优视觉版本

```powershell
python visual_alpha_sweep_eval_series_rules.py --report output_128/report.json --photo pic_all.jpg --alpha-list "1.0" --background-mode original --bg-scale 0 --use-visual-series-rules --rule-strength 0.7
```

### 11.4 如果没有自动找到画圆文件

手动指定：

```powershell
python visual_alpha_sweep_eval_series_rules.py --report output_128/report.json --photo pic_all.jpg --alpha-list "1.0" --background-mode original --bg-scale 0 --use-visual-series-rules --rule-strength 0.7 --visual-circle-file output_128/corrected_residual_alpha_sweep/visual_circles_manual.json
```

---

## 12. 可调参数建议

当前最佳值：

```text
alpha = 1.0
rule_strength = 0.7
background_mode = original
bg_scale = 0
```

如果后续需要微调，可以按以下方向调整。

### 12.1 颜色仍然偏过修

表现：

```text
偏浅
偏黄
偏白
```

尝试：

```powershell
--rule-strength 0.8
--rule-strength 1.0
```

规则强度越大，对过修的保护越强。

### 12.2 颜色看起来修正不够

表现：

```text
太接近原图
标准色靠近不明显
```

尝试：

```powershell
--rule-strength 0.5
```

规则强度越小，越接近完整 residual 修正。

### 12.3 背景过于原始

表现：

```text
背景仍有明显偏黄、偏暗
```

可以尝试轻微混合 corrected 背景：

```powershell
--background-mode blend --background-mix 0.10
```

不建议一开始超过 0.20。

### 12.4 胶块边缘过硬

尝试提高 feather：

```powershell
--feather 41
--feather 51
```

但 feather 太大可能影响邻近背景。

---

## 13. 注意事项

### 13.1 当前版本适合已知 128 色板

当前 residual 依赖已知标准 Lab 和对应色号，因此适合：

```text
已知 128 色板
实验验证
视觉展示
算法调参
```

如果用于未知胶块识别场景，需要将目标 Lab 从真实标签标准值改为：

```text
算法识别出的 Top1 / TopK 标准值
```

并结合置信度判断是否吸附。

### 13.2 手动画圆文件必须备份

```text
visual_circles_manual.json
```

是人工成本最高的文件。建议和最终图一起备份。

### 13.3 当前图像是视觉结果，不等同于原始测色结果

最终 preview 图是为了肉眼效果优化的展示图。

正式测色数据仍应以：

```text
report.json
06_target_validation.csv
```

为准。

---

## 14. 当前版本总结

当前版本可以命名为：

```text
corrected residual + manual circle + original background + series visual rules v0.7
```

中文描述：

```text
在 ColorChecker 校正基础上，对 128 个胶块计算标准 Lab residual；
通过手动画圆限定胶块视觉区域；
背景保留原始照片，避免校正过度；
对容易过修的色系加入分组保护规则；
最终使用 rule_strength=0.7，在标准色靠近与肉眼自然之间取得较好平衡。
```
