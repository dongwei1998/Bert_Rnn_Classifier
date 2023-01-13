## Bert+RNN实现的文本分类
+ 需要下载预训练模型  [chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
+ 自行修改文件配置路径 /bert_utils/parameter.py  文件
### 介绍
+ 本项目通过 BERT模型拼接 RNN模型实现
+ 可以完后文本分类、情感分析、等任务

### 数据集
+ 数据集采用的是搜狐新闻文本
+ 格式为 csv  包含三个文件，训练、验证、测试
+ 数据内部结构为 label text格式，一个一条数据
  

### 环境

+ python 3.8.1
+ requests==2.26.0
+ Flask==1.1.2
+ tensorflow==1.14.0
+ openpyxl==3.0.7
+ six==1.16.0
+ numpy==1.20.3

### 训练
    python run_train.py
### 服务启动
    demo文件夹下
    python deploy_with_web_api.py
### 服务测试
    demo文件夹下
    python wep_api_demo.py
