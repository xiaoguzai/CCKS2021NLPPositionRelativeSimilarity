import json
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
with open('/home/xiaoguzai/数据/天池比赛地址相关数据集/Xeon3NLP_round1_train_20210524.json','r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict[0:10])
bert_ckpt_dir="/home/xiaoguzai/下载/chinese_L-12_H-768_A-12/"
bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
bert_config_file = bert_ckpt_dir + "bert_config.json"
from tokenization import FullTokenizer
from tqdm import tqdm
import os
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir,"vocab.txt"))
question1_id,question2_id = [],[]
question1_segment,question2_segment = [],[]
label_id = []
for data in load_dict:
    text1 = data['query']
    token1 = tokenizer.tokenize(text1)
    token1 = ["[CLS]"]+token1+["[SEP]"]
    token_id1 = tokenizer.convert_tokens_to_ids(token1)
    for data1 in data['candidate']:
        text2 = data1['text']
        token2 = tokenizer.tokenize(text2)
        token2 = ["[CLS]"]+token2+["[SEP]"]
        token_id2 = tokenizer.convert_tokens_to_ids(token2)
        question1_id.append(token_id1)
        question2_id.append(token_id2)
        question1_segment.append([0]*len(token_id1))
        question2_segment.append([1]*len(token_id2))
        if data1['label'] == '不匹配':
            label_id.append(0)
        elif data1['label'] == '部分匹配':
            label_id.append(1)
        else:
            label_id.append(2)
import models
from models import Bert
batch_size = 48
max_seq_len = 128
bertmodel = Bert(maxlen=max_seq_len,batch_size=batch_size)
input_ids = [[keras.layers.Input(shape=(None,),dtype='int32',name="token_ids1"),
            keras.layers.Input(shape=(None,),dtype='int32',name="segment_ids1")],
            [keras.layers.Input(shape=(None,),dtype='int32',name="token_ids2"),
            keras.layers.Input(shape=(None,),dtype='int32',name="segment_ids2")]]
output1 = bertmodel(input_ids[0])
output2 = bertmodel(input_ids[1])
output1 = keras.layers.Lambda(lambda seq: seq[:,0,:])(output1)
output2 = keras.layers.Lambda(lambda seq: seq[:,0,:])(output2)
#!!!这里取出0而不取出-1是因为最后一位可能有填充数值
output = K.concatenate([output1,output2],axis=-1)
output = keras.layers.Dropout(0.5)(output)
output = keras.layers.Dense(units=768,activation="tanh")(output)
output = keras.layers.Dropout(0.5)(output)
output = keras.layers.Dense(units=3,activation="softmax")(output)
model = keras.Model(inputs=input_ids,outputs=output)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
#from_logits = False输入已经符合某种分布，系统只会把你概率归一化
#from_logits = True输入的是原始数据，系统会帮你softmax之后再进行计算
model.summary()

from loader import load_stock_weights
load_stock_weights(bert=bertmodel,ckpt_path=bert_ckpt_file)

class DataGenerator(object):
    def __init__(self,token_ids1,segment_ids1,token_ids2,segment_ids2,label_ids,batch_size=48,max_seq_len=128):
        self.token_ids1 = token_ids1
        self.segment_ids1 = segment_ids1
        self.token_ids2 = token_ids2
        self.segment_ids2 = segment_ids2
        self.batch_size = batch_size
        self.maxlen = max_seq_len
        self.totals = len(self.token_ids1)
        self.label_ids = label_ids
    
    def __len__(self):
        return int(np.floor(len(self.token_ids1)/self.batch_size))

    def sample(self,random=False):
        indices = list(range(len(self.token_ids1)))
        np.random.shuffle(indices)
        for i in indices:
            yield self.token_ids1[i],self.segment_ids1[i],self.token_ids2[i],self.segment_ids2[i],self.label_ids[i]
    
    def __iter__(self,random=False):
        random = False
        batch_token_ids1,batch_segment_ids1 = [],[]
        batch_token_ids2,batch_segment_ids2 = [],[]
        batch_label_ids = []
        currents = 0
        batch_data = []
        for token_ids1,segment_ids1,token_ids2,segment_ids2,label_ids in self.sample(random):
            if len(token_ids1) > self.maxlen:
                token_ids1 = token_ids1[:self.maxlen]
                segment_ids1 = segment_ids1[:self.maxlen]
            if len(token_ids2) > self.maxlen:
                token_ids2 = token_ids2[:self.maxlen]
                segment_ids2 = segment_ids2[:self.maxlen]
            batch_token_ids1.append(token_ids1)
            batch_segment_ids1.append(segment_ids1)
            batch_token_ids2.append(token_ids2)
            batch_segment_ids2.append(segment_ids2)
            batch_label_ids.append(label_ids)
            currents = currents+1
            if len(batch_token_ids1) == self.batch_size or currents == self.totals:
                batch_token_ids1 = sequence_padding(batch_token_ids1)
                batch_segment_ids1 = sequence_padding(batch_segment_ids1)
                batch_token_ids2 = sequence_padding(batch_token_ids2)
                batch_segment_ids2 = sequence_padding(batch_segment_ids2)
                yield [[np.array(batch_token_ids1),np.array(batch_segment_ids1)],[np.array(batch_token_ids2),np.array(batch_segment_ids2)]],np.array(batch_label_ids)
                #yield [np.array(batch_token_ids1),np.array(batch_segment_ids1)],np.array(batch_label_ids)
                batch_token_ids1,batch_segment_ids1,batch_token_ids2,batch_segment_ids2,batch_label_ids = [],[],[],[],[]
                #这里必须要统一维度
                
    def cycle(self,random=True):
        while True:
            for d in self.__iter__(random):
                yield d

def sequence_padding(inputs,padding = 0):
    length = max([len(x) for x in inputs])
    pad_width = [(0,0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0,length-len(x))
        x = np.pad(x,pad_width,'constant',constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

model.save_weights('./天池地址匹配new.h5')

with open('/home/xiaoguzai/数据/天池比赛地址相关数据集/Xeon3NLP_round1_test_20210524.json','r') as load_f:
    test_data = json.load(load_f)

#for data in test_data:
for i in tqdm(range(0,len(test_data))):
    data = test_data[i]
    text1 = data['query']
    token1 = tokenizer.tokenize(text1)
    token1 = ["[CLS]"]+token1+["[SEP]"]
    token_id1 = tokenizer.convert_tokens_to_ids(token1)
    segment_id1 = [0]*len(token_id1)
    for j in range(len(data['candidate'])):
        data1 = data['candidate'][j]
        text2 = data1['text']
        token2 = tokenizer.tokenize(text2)
        token2 = ["[CLS]"]+token2+["[SEP]"]
        token_id2 = tokenizer.convert_tokens_to_ids(token2)
        segment_id2 = [1]*len(token_id2)
        res = model.predict([[np.array([token_id1]),np.array([segment_id1])],[np.array([token_id2]),np.array([segment_id2])]]).argmax(axis=-1)
        if res[0] == 0:
            test_data[i]['candidate'][j]['label'] = '不匹配'
        elif res[0] == 1:
            test_data[i]['candidate'][j]['label'] = '部分匹配'
        else:
            test_data[i]['candidate'][j]['label'] = '完全匹配'

file_obj = open('/home/xiaoguzai/数据/天池比赛地址相关数据集/results2.txt','w',encoding='utf-8')
for data in test_data:
    file_obj.write(json.dumps(data,ensure_ascii=False))
    file_obj.write('\n')
file_obj.close()