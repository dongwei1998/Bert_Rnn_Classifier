# -*- coding:utf-8 -*-
# @Time      : 2021-08-01 18:42
# @Author    : 年少无为呀！
# @FileName  : parameter.py
# @Software  : PyCharm

import argparse

import os
# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
# 获取项目更目录
root_directory = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")

# 命令行参数定义和解析
parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument('--prob_cache_file', type = str,default=os.path.join(root_directory,'demo/cache_file'),help='API预测缓存文件')
parser.add_argument('--logini', type = str,default=os.path.join(root_directory,'config/logging.ini'),help='日志配置文件')
parser.add_argument('--do_train', type=bool, default=True, help='是否训练')
parser.add_argument('--do_eval', type=bool, default=True, help='是否验证')
parser.add_argument('--do_predict', type=bool, default=False, help='是否预测')
parser.add_argument('--max_seq_length', type = int,default=512,help='序列的最大长度，最长不要超过bert长度512')
parser.add_argument('--label_list', type = list,default=['胜诉','败诉','撤诉','其他'],help='训练数据中的实际的label标签值')
parser.add_argument('--task_name', type = str,default='zhfw',help='训练任务的名称，对应数据预处理')

parser.add_argument('--data_dir', type = str,default=os.path.join(root_directory,'data/falv_files'),help='训练的数据')
parser.add_argument('--train_batch_size', type = int,default=2,help='训练批次大小')
parser.add_argument('--eval_batch_size', type = int,default=8,help='验证批次大小')
parser.add_argument('--predict_batch_size', type = int,default=8,help='预测批次大小')
parser.add_argument('--num_train_epochs', type = int,default=1,help='训练总批次大小')
parser.add_argument('--learning_rate', type = float,default=1e-5,help='训练学习率大小')
parser.add_argument('--save_checkpoints_steps', type = int,default=20,help='多少步保存一次模型')
parser.add_argument('--iterations_per_loop', type = int,default=50,help='多少步评估一次模型')
parser.add_argument('--warmup_proportion', type = float,
                    default=0.1,
                    help='warmup_proportion表示，慢热学习的比例。比如warmup_proportion=0.1，总步数=100，那么warmup步数就为10。在1到10步中，学习率会比10')


# 是否 使用 TPU
parser.add_argument('--use_tpu', type=bool, default=False, help='是否使用tpu')
parser.add_argument('--tpu_name', type=str, default=None, help='使用的tpu的名称')
parser.add_argument('--tpu_zone', type=str,
                    default=None,
                    help='[可选]Cloud TPU所在的GCE区域。 如果没有指定，我们将尝试从元数据中自动检测GCE项目。 ')
parser.add_argument('--gcp_project', type=str,
                    default=None,
                    help='[可选]Cloud TPU所在的GCE区域。 如果没有指定，我们将尝试从元数据中自动检测GCE项目。 ')
parser.add_argument('--master', type=str, default=None, help='[可选]TensorFlow主URL。')
parser.add_argument('--num_tpu_cores', type = int,default=4,help='仅当use_tpu为True时使用。 TPU核总数。  ')



# bert模型配置
parser.add_argument('--do_lower_case', type = bool,default=True,help='是否小写')
parser.add_argument('--init_checkpoint', type = str,
                    default=os.path.join(root_directory,'modelParams/chinese_L-12_H-768_A-12/bert_model.ckpt'),
                    help='bert初始检查点，存放bert原始模型')
parser.add_argument('--vocab_file', type = str,
                    default=os.path.join(root_directory,'modelParams/chinese_L-12_H-768_A-12/vocab.txt'),
                    help='训练BERT模型的词汇表文件')
parser.add_argument('--bert_config_file', type = str,
                    default=os.path.join(root_directory,'modelParams/chinese_L-12_H-768_A-12/bert_config.json'),
                    help='训练bert模型的配置文件')
parser.add_argument('--output_dir', type = str,
                    default=os.path.join(root_directory,'model/falv'),
                    help='模型最终保存以及可视化文件夹')
# 数据格式转换配置
parser.add_argument('--org_files_xlsx', type=str,
                    default=os.path.join(root_directory,"original_data/知识库增强数据_20210603.xlsx"),
                    help='原始训练数据')
parser.add_argument('--train_csv', type = str,
                    default=os.path.join(root_directory,"data/falv_files/train.csv"),
                    help='生成训练数据')
parser.add_argument('--test_csv', type = str,
                    default=os.path.join(root_directory,"data/falv_files/test.csv"),
                    help='生成测试数据')
parser.add_argument('--dev_csv', type = str,
                    default=os.path.join(root_directory,"data/falv_files/dev.csv"),
                    help='生成以验证数据')













# 配置日志
args = parser.parse_args()
# print(args.logini)






