# -*- coding: utf-8 -*-

import numpy as np
import paddle
from paddle.io import DataLoader
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator
from utils import convert_example, evaluate, predict, load_dict
from functools import partial
import xlrd
import pandas as pd
#打开excel

filename = 'detailAddress.xlsx'
def read_excel1(filename):
    wb = xlrd.open_workbook(filename)
    #按工作簿定位工作表
    sh = wb.sheet_by_name('Sheet3')
    # print(sh.nrows)#有效数据行数
    # print(sh.ncols)#有效数据列数
    # print(sh.cell(0,0).value)#输出第一行第一列的值
    # print(sh.row_values(0))#输出第一行的所有值
    #将数据和标题组合成字典
    #print(dict(zip(sh.row_values(0),sh.row_values(1))))
    #遍历excel，打印所有数据
    jd = []
    jdmc = []
    jd_address = []
    for i in range(1,sh.nrows):
    #for i in range(1,20):
        if  sh.row_values(i)[2]!="":
            print(sh.row_values(i))
            jd.append(sh.row_values(i)[0])
            jdmc.append(sh.row_values(i)[1])
            jd_address.append(sh.row_values(i)[2])
    #df = pd.DataFrame(np.arange(20).reshape(4, 5), columns=['d1', 'd2', 'd3', 'd4', 'd5'])
    dict = {"jd":jd,"jdmc":jdmc,"jd_address":jd_address}
    jd_df = pd.DataFrame(dict)
    print(jd_df.iloc[5,:])
    return jd,jdmc,jd_address

#!head -n20 dataset/final_test.txt

# #定义test数据集
def load_dataset(datafiles):
    def read_excel(filename):
        wb = xlrd.open_workbook(filename)
        # 按工作簿定位工作表
        sh = wb.sheet_by_name('Sheet3')
        for i in range(1,sh.nrows):

            if sh.row_values(i)[2]!="":
                words = list(sh.row_values(i)[2])
                words = [ch for ch in words]
                #print("words",words)
                # 要预测的数据集没有label，伪造个O，不知道可以不 ，应该后面预测不会用label
                labels = ['O' for x in range(0, len(words))]
                #print("labels", labels)

                yield words, labels
                # yield words

    if isinstance(datafiles, str):
        return MapDataset(list(read_excel(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read_excel(datafile))) for datafile in datafiles]

    # def read(data_path):
    #     with open(data_path, 'r', encoding='utf-8') as fp:
    #         # next(fp)  # 没有header，不用Skip header
    #         for line in fp.readlines():
    #             ids, words = line.strip('\n').split('\001')
    #             #words = list(jdaddress)
    #             #words[-1]#去掉最后一个"\n"
    #             print("ids,words",words)
    #             words=[ch for ch in words]
    #             # 要预测的数据集没有label，伪造个O，不知道可以不 ，应该后面预测不会用label
    #             labels=['O' for x in range(0,len(words))]
    #             print("labels",labels)
    #
    #             yield words, labels
    #             # yield words
    #
    # if isinstance(datafiles, str):
    #     return MapDataset(list(read(datafiles)))
    # elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
    #     return [MapDataset(list(read(datafile))) for datafile in datafiles]

import pandas as pd

# df = pd.read_excel('detailAddress.xlsx')		# 使用pandas模块读取数据
# df1 = pd.read_excel('detailAddress.xlsx')		# 使用pandas模块读取数据
# df2 = pd.read_excel('detailAddress.xlsx')		# 使用pandas模块读取数据
#
# print('开始写入txt文件...')
# df.to_csv('detailAddress.txt', header=False, sep=',', index=False)		# 写入，逗号分隔
# df.to_csv('detailAddress1.txt', header=False, sep=',', index=False)		# 写入，逗号分隔
# df.to_csv('detailAddress2.txt', header=False, sep=',', index=False)		# 写入，逗号分隔
#
# print('文件写入成功!')

# Create dataset, tokenizer and dataloader.

test_ds = load_dataset(datafiles=('./detailAddress.xlsx'))

# for i in range(10):
#     print("22222",test_ds1[i])
# # Create dataset, tokenizer and dataloader.

#test_ds = load_dataset(datafiles=('./dataset/final_test1.txt'))

# import ipdb
# ipdb.set_trace


for i in range(2):
    print("1111",test_ds[i])

#加载训练好的模型
label_vocab = load_dict('./dataset/mytag.dic')
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)
test_ds.map(trans_func)
print ("0000000",test_ds[0])

ignore_label = 1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
    Pad(axis=0, pad_val=ignore_label)  # labels
): fn(samples)


test_loader = paddle.io.DataLoader(
    dataset=test_ds,
    batch_size=30,
    return_list=True,
    collate_fn=batchify_fn)

def my_predict(model, data_loader, ds, label_vocab):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        # print(len(logits[0]))
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds ,tags= parse_decodes(ds, pred_list, len_list, label_vocab)
    return preds, tags

# Define the model netword and its loss
model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))

model_dict = paddle.load('ernie_result/model_state.pdparams')
model.set_dict(model_dict)

#预测并保存
from utils import *
preds, tags = my_predict(model, test_loader, test_ds, label_vocab)



file_path = "results1.txt"
# file_path = "./dataset/final_test1.txt"
jd,jdmc,jd_address = read_excel1('detailAddress.xlsx')


def result_deal(jd_address,preds):
    Dlen = len(preds)
    town_road_list = []
    town_road_other_list = []
    for i  in range(Dlen):
        #print("item",type(item),item)
        item = preds[i].split(" ")
        print("itemlen",len(item),item)
        print(len(list(jd_address[i])),list(jd_address[i]))
        if  "E-road"  in item:
            print("iE-road",item.index("E-road"))
            split_index = item.index("E-road")
            town_road_list.append("".join(list(jd_address[i])[:split_index]))
            town_road_other_list.append("".join(list(jd_address[i])[split_index:]))
        elif "E-town" in item:
            split_index = item.index("E-town")
            town_road_list.append("".join(list(jd_address[i])[:split_index]))
            town_road_other_list.append("".join(list(jd_address[i])[split_index:]))
        else:
            split_index = 0
            town_road_list.append("")
            town_road_other_list.append("".join(list(jd_address[i])))
        print("分列坐标",split_index)
        print("街道","".join(list(jd_address[i])[:split_index]),"\n","rest","".join(list(jd_address[i])[split_index:]))
        # town_road_list.append("".join(list(jd_address[i])[:split_index+1]))
        # town_road_other_list.append("".join(list(jd_address[i])[split_index:]))
        # if  "E-road" or "E-town" in item:
        #     town_road_list
    return town_road_list, town_road_other_list

town_road_list, town_road_other_list = result_deal(jd_address,preds)

dict = {"jd":jd,"jdmc":jdmc,"jd_address":jd_address,"road_town":town_road_list,"the_rest": town_road_other_list}
jd_df = pd.DataFrame(dict)
jd_df.to_csv("road_result222.csv")

# with open(file_path, "w", encoding="utf8") as fout:
#     print("preds",type(preds),preds)
#     print("tags",type(tags),tags)
#     fout.write("\n".join(preds))
#     for item in preds:
        #fout.write(item)
#Print some examples
print(
    "The results have been saved in the file: %s, some examples are shown below: "
    % file_path)

def main():
    data_list = []
    with open('ernie_results1.txt', encoding='utf-8') as f:
        data_list = f.readlines()
    print("data_list")
    return data_list



#转换保存结果

# if __name__ == "__main__":
#     # print('1^ A浙江杭州阿里^AB-prov E-prov B-city E-city B-poi E-poi')
#     sentence_list = main()
    # print(len(sentence_list))

    # final_test = []
    # with open('dataset/final_test1.txt', encoding='utf-8') as f:
    #     final_test = f.readlines()
    # test_data = []
    # print(f'{len(final_test)}\t\t{len(sentence_list)}')
    # for i in range(len(final_test)):
    #     # test_data.append(final_test[i].strip('\n') + '\001' + sentence_list[i] + '\n')
    #     print("sentence_list[i]",sentence_list[i])
    #     test_data.append(final_test[i].strip('\n').strip(' ') + '\001' + sentence_list[i].strip(' '))
    # with open('predict1.txt', 'w', encoding='utf-8') as f:
    #     f.writelines(test_data)
    # print(50 * '*')
    # print('write result ok!')
    # print(50 * '*')
