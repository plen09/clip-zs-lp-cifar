# CLIP Zero-Shot vs Prompt Engineering vs Linear Probe on CIFAR-10

本项目比较 OpenAI CLIP ViT-B/32 在 CIFAR-10 上的四类设置：

1. Zero-shot：仅使用类别名；
2. Single Prompt Engineering：测试多种文本模板；
3. Prompt Ensemble：对多个模板的文本特征进行集成；
4. Linear Probe：冻结 CLIP image encoder，仅训练线性分类器。

项目核心目标是回答：CLIP 的视觉特征有多强、Prompt 对 Zero-shot 分类有多敏感，以及一个极轻量的监督线性分类器能否显著超过文本提示驱动的 Zero-shot 方法。

## 1. Experimental Setup

- **Dataset**: CIFAR-10
- **Train set**: 50,000 images
- **Test set**: 10,000 images
- **Classes**: 10
- **Backbone**: OpenAI CLIP ViT-B/32
- **Text/Image embedding dimension**: 512
- **Image preprocessing**: CLIP-provided `preprocess`
- **Zero-shot classifier**: cosine similarity + `argmax`
- **Linear Probe**: frozen CLIP image encoder + `nn.Linear(512, 10)`
- **Linear Probe trainable parameters**: 5,130

本项目保持同一 CLIP backbone、同一 CIFAR-10 数据集和同一 CLIP preprocessing，避免因为更换模型或预处理方式引入额外变量。

## 2. Experimental Process

详见 `main` 下的 `report.pdf`。

## 3. Conclusions

本实验得到以下主要结论：

1. **CLIP 在 CIFAR-10 上具有很强的 Zero-shot 能力**：仅使用类别名即可达到 87.38%。
2. **Prompt wording 会显著影响分类结果**：最佳整体单模板达到 89.03%，但不同类别对 prompt 的响应差异非常大。
3. **更具体的 prompt 不一定更好**：`"a close-up photo of a {}"` 总体表现甚至低于 class-name baseline。
4. **Prompt Ensemble 的主要价值是鲁棒性，而不是最高准确率**：它达到 88.87%，几乎追平最佳单 prompt，并明显优于成员 prompt 的平均表现。
5. **冻结的 CLIP visual representation 包含比 Zero-shot 文本对齐所利用的更多类别信息**：只训练 5,130 个参数的 Linear Probe 即达到 93.76%。
6. **Linear Probe 的收益具有明显类别依赖性**：`frog`、`airplane`、`cat`、`deer` 提升很大，而 `horse`、`bird` 并未超过 Zero-shot。
7. **Zero-shot 与 Linear Probe 的差距说明，视觉表征能力和跨模态文本对齐能力不是同一件事**。CLIP image encoder 已经学习到强判别特征，但文本模板未必能为每个类别提供最合适的决策边界。

## 4. Project Outputs

项目目录的结构如下：

```text
outputs/
├── figures/
│   ├── overall_accuracy_comparison.png
│   ├── per_class_accuracy_comparison.png
│   ├── confusion_matrix_zero_shot.png
│   └── confusion_matrix_linear_probe.png
│
├── models/
│   └── linear_probe_cifar10.pt
│
└── results/
    ├── zero_shot_results.csv
    ├── per_class_accuracy.csv
    ├── confusion_matrix_zero_shot_class_name.csv
    ├── single_prompt_overall_ranking.csv
    ├── prompt_sensitivity_by_class.csv
    ├── prompt_ensemble_overall.csv
    ├── prompt_ensemble_per_class_accuracy.csv
    ├── confusion_matrix_prompt_ensemble.csv
    ├── prompt_ensemble_class_analysis.csv
    ├── linear_probe_overall.csv
    ├── linear_probe_per_class_accuracy.csv
    ├── confusion_matrix_linear_probe.csv
    ├── linear_probe_training_history.csv
    ├── final_method_comparison.csv
    ├── final_per_class_comparison.csv
    ├── final_class_level_analysis.csv
    ├── top_confusions_zero_shot.csv
    └── top_confusions_linear_probe.csv
```

## 5. Reproducibility Note

Linear Probe 的 CLIP visual feature extraction 最终采用：

- fresh kernel；
- same local OpenAI CLIP ViT-B/32 checkpoint；
- FP32 visual encoder；
- frozen CLIP parameters；
- normalized image features；
- feature extraction batch size 64。

这样可以避免此前长时间 CUDA session 中出现的非有限特征和非重复 forward 问题，同时保持模型、权重和 preprocessing 不变。
