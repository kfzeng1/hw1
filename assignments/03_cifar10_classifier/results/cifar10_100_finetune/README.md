# CIFAR-10 100轮微调结果

训练方式：从 50 轮 checkpoint 继续微调 50 轮，总轮数 100。
学习率：继续训练使用 base lr=0.012、warmup=0，按 100 轮 cosine schedule 从约 0.006 逐步衰减。

最佳测试准确率：97.30%（第 96 轮）
第 100 轮测试准确率：97.25%
独立评估 accuracy：97.30%
macro F1：97.30%
weighted F1：97.30%
测试集正确数：9730 / 10000

checkpoint 保存在本地 checkpoints_100_finetune/，不上传 GitHub。
