## “合肥高新杯”心电人机智能大赛——心电异常事件预测    rank 17th solution

来源：["合肥高新杯"心电人机智能大赛](https://tianchi.aliyun.com/competition/entrance/231754/introduction)

**round2 Score=0.92696**

系统环境：*ubuntu python3.6 pytorch:latest*

**大致思路：每个导联作为一个通道，使用1维卷积进行训练**

## 1、数据预处理
采用wfdb库QRS波提取每个导联心拍，然后对每个样本每个导联计算所有心拍median。data_process.py 对训练和验证集进行划分，beat_process.py 对心拍进行提取，
```shell
python round1_data_process.py 

python round1_beat_process.py

python round2_data_process.py

python round2_beat_process.py
```

## 2、模型训练
采用round1数据进行模型预训练，然后在round2数据上进行fine-tune.
```shell
python main.py train --model_name=mixnet_sm_pretrain

python main.py train_cv --model_name=mixnet_sm

python main.py train --model_name=mixnet_mm_pretrain

python main.py train_cv --model_name=mixnet_mm

python main.py val_cv --ckpt=ckpt/mixnet_sm_cv --model_name=mixnet_sm_predict

python main.py val_cv --ckpt=ckpt/mixnet_mm_cv --model_name=mixnet_mm_predict
```

## 3、模型测试
模型测试，在submit文件夹下生成提交结果
```shell
python main.py test_cv_ensemble --ckpt=ckpt #加载训练权重进行测试
```

**一些细节**

 1. 本次测试模型为1d mixnet;
 2. 训练数据没有只进行了简单的数据增强.


参考:
[baseline](https://github.com/JavisPeng/ecg_pytorch)
[答辩视频](https://tianchi.aliyun.com/course/video?liveId=41127)
[top solution](https://tianchi.aliyun.com/competition/entrance/231754/forum)

