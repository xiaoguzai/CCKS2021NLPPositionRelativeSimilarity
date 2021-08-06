import json
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import jieba
import numpy as np
import time
start = time.time()
with open('/赛道三/tcdata/round1_train.json','r') as load_f:
    load_dict = json.load(load_f)

with open('/赛道三/tcdata/round2_train.json','r') as load_f:
    load_dict = load_dict+json.load(load_f)

with open('/赛道三/tcdata/Xeon3NLP_round1_test_20210524.json','r') as load_f:
    load_dict = load_dict+json.load(load_f)

f = open('/tcdata/test2_release.txt')
line = f.readline()
data = ['[']
while line:
    data.append(line)
    line = f.readline()
f.close()
data.append(']')

f = open('/赛道三/tcdata/test2_release.json','w')
for index in range(len(data)):
    if index == 0:
        f.write(data[index]+'\n')
        #[+\n
    elif index == len(data)-2:
        f.write(data[index])
    elif index == len(data)-1:
        #']'
        f.write(data[index])
    else:
        f.write(data[index][:-1]+','+'\n')
f.close()

with open('/赛道三/tcdata/test2_release.json','r') as load_f:
    load_dict = load_dict+json.load(load_f)

nezha_ckpt_dir="/赛道三/tcdata/NEZHA-Base-WWM/"
nezha_ckpt_file = nezha_ckpt_dir + "model.ckpt-691689"
nezha_config_file = nezha_ckpt_dir + "bert_config.json"

from tokenization import FullTokenizer
from tqdm import tqdm
import os
tokenizer = FullTokenizer(vocab_file=os.path.join(nezha_ckpt_dir,"vocab.txt"))
file_name = '/赛道三/tcdata/NEZHA-Base-WWM/vocab.txt'
myfile = open(file_name)
vocab_size = len(myfile.readlines())
question1_id,question2_id = [],[]
question1_segment,question2_segment = [],[]
question_segment = []
segment_id = []
question_text = []
label_id = []

zero_number = 0
first_number = 0
second_number = 0
max_seq_len = 128
question_id = []
pos_id = []
for data in load_dict:
    text1 = data['query']
    token1 = tokenizer.tokenize(text1)
    token1 = ["[CLS]"]+token1
    token_id1 = tokenizer.convert_tokens_to_ids(token1)
    if len(token_id1) > max_seq_len-1:
        token_id1 = token_id1[:max_seq_len-1]
    token_id1 = token_id1+tokenizer.convert_tokens_to_ids(["[SEP]"])
    question1_segment = [0]*len(token_id1)
    for data1 in data['candidate']:
        text2 = data1['text']
        token2 = tokenizer.tokenize(text2)
        token2 = token2
        token_id2 = tokenizer.convert_tokens_to_ids(token2)
        if len(token_id2) > max_seq_len-1:
            token_id2 = token_id2[:max_seq_len-1]
        token_id2 = token_id2+tokenizer.convert_tokens_to_ids(["[SEP]"])
        question2_segment = [1]*len(token_id2)
        token_id = token_id1+token_id2
        question_segment = question1_segment+question2_segment
        segment_id.append(question_segment)
        question_id.append(token_id)
        question_text.append('$'+text1+'$'+text2+'$')
        #question_text正好匹配对应的token_id位置的内容
        label_id.append(np.random.randint(0,5))

mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])
mask_id = mask_id[0]

question_segment_text = []
question_segment_pos = []
#传入的参数设为question_text,question_id,segment_id
from nezha_pretraining_data_process import nezha_pretraining_store_data
question_new_id,question_new_segment = nezha_pretraining_store_data(question_id,segment_id,question_text,mask_id,vocab_size)
#nezha_pretraining_get_new_data()
question_label_id = [np.random.randint(0,3) for _ in range(len(question_new_id))]

with open(nezha_config_file,'r') as load_f:
    config = json.load(load_f)
print('config = ')
print(config)
config['embedding_size'] = config['hidden_size']
config['num_layers'] = config['num_hidden_layers']

from nezha import Bert
bertmodel = Bert(maxlen=max_seq_len,batch_size=128,with_mlm=True,**config)
input_ids1 = keras.layers.Input(shape=(None,),dtype='int32',name="token_ids1")
input_ids2 = keras.layers.Input(shape=(None,),dtype='int32',name="segment_ids1")
#这里不能使用CrossEntropy的原因在于label_id之中无法取出每一波的批次值，所以选择使用loss函数
output_ids = bertmodel([input_ids1,input_ids2])
model = keras.Model(inputs=[input_ids1,input_ids2],outputs=output_ids)
model.summary()

from nezha_loader import load_nezha_stock_weights
load_nezha_stock_weights(nezha=bertmodel,ckpt_path=nezha_ckpt_file)

from sklearn.model_selection import KFold
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
result_point = []
test_point = []
class DataGenerator(keras.callbacks.Callback):
    def __init__(self,token_ids,segment_ids,label_ids,batch_size=128,max_seq_len=128):
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.batch_size = batch_size
        self.maxlen = max_seq_len
        self.totals = len(self.token_ids)
    
    def __len__(self):
        return int(np.floor(len(self.token_ids)/self.batch_size))

    def sample(self,random=False):
        #sample拆分出训练集进行训练
        indices = list(range(len(self.token_ids)))
        np.random.shuffle(indices)
        for i in indices:
            yield self.token_ids[i],self.segment_ids[i]
    
    def __iter__(self,random=False):
        #__iter__拆分出对应的batch
        random = False
        batch_token_ids = []
        batch_segment_ids = []
        batch_label_ids = []
        currents = 0
        batch_data = []
        for token_ids,segment_ids in self.sample(random):
            if len(token_ids) > self.maxlen*2:
            #!!!注意这里要为self.maxlen*2,因为单个长度为128，两个加起来不能超过256
                token_ids = token_ids[:self.maxlen]
                segment_ids = segment_ids[:self.maxlen]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            currents = currents+1
            if len(batch_token_ids) == self.batch_size or currents == self.totals:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [np.array(batch_token_ids),np.array(batch_segment_ids)],np.array(batch_token_ids)
                #yield [np.array(batch_token_ids1),np.array(batch_segment_ids1)],np.array(batch_label_ids)
                batch_token_ids,batch_segment_ids= [],[]
                #这里必须要统一维度
                
    def cycle(self,random=True):
        while True:
            for d in self.__iter__(random):
                yield d
    
    def on_epoch_end(self,epoch,logs):
        model.save_weights('/赛道三/tcdata/预训练天池地址匹配the_best_model.h5')
        #注意这里存的为epoch+5

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

train_generator = DataGenerator(question_new_id,question_new_segment,question_label_id,batch_size=128)
model.compile(#optimizer=keras.optimizers.Adam(learning_rate=2e-5),
              optimizer=keras.optimizers.Adam(learning_rate=2e-5),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              #loss=compute_loss,
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
#修改2
model.fit(
    train_generator.cycle(),
    steps_per_epoch = len(train_generator),
    epochs = 2,
    callbacks = [train_generator]
)

#修改3
#model.save_weights('/赛道三/tcdata/预训练天池地址匹配the_best_model.h5')
import json
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
with open('/赛道三/tcdata/round1_train.json','r') as load_f:
    load_dict = json.load(load_f)

with open('/赛道三/tcdata/round2_train.json','r') as load_f:
    load_dict = load_dict+json.load(load_f)

nezha_ckpt_dir="/赛道三/tcdata/NEZHA-Base-WWM/"
nezha_ckpt_file = nezha_ckpt_dir + "model.ckpt-691689"
nezha_config_file = nezha_ckpt_dir + "bert_config.json"

from tokenization import FullTokenizer
from tqdm import tqdm
import os
tokenizer = FullTokenizer(vocab_file=os.path.join(nezha_ckpt_dir,"vocab.txt"))
question1_id,question2_id = [],[]
question1_segment,question2_segment = [],[]
question_segment = []
segment_id = []
question_text = []
label_id = []

zero_number = 0
first_number = 0
second_number = 0
max_seq_len = 128
question_id = []
for data in load_dict:
    text1 = data['query']
    token1 = tokenizer.tokenize(text1)
    token1 = ["[CLS]"]+token1
    token_id1 = tokenizer.convert_tokens_to_ids(token1)
    if len(token_id1) > max_seq_len-1:
        token_id1 = token_id1[:max_seq_len-1]
    token_id1 = token_id1+tokenizer.convert_tokens_to_ids(["[SEP]"])
    question1_segment = [0]*len(token_id1)
    for data1 in data['candidate']:
        text2 = data1['text']
        token2 = tokenizer.tokenize(text2)
        token2 = token2
        token_id2 = tokenizer.convert_tokens_to_ids(token2)
        if len(token_id2) > max_seq_len-1:
            token_id2 = token_id2[:max_seq_len-1]
        token_id2 = token_id2+tokenizer.convert_tokens_to_ids(["[SEP]"])
        question2_segment = [1]*len(token_id2)
        token_id = token_id1+token_id2
        question_segment = question1_segment+question2_segment
        segment_id.append(question_segment)
        question_id.append(token_id)
        question_text.append(text1+' '+text2)
        if data1['label'] == '不匹配':
            label_id.append(0)
            zero_number = zero_number+1
        elif data1['label'] == '部分匹配':
            label_id.append(1)
            first_number = first_number+1
        else:
            label_id.append(2)
            second_number = second_number+1
            
            
            label_id.append(2)
            segment_id.append(question_segment)
            question_id.append(token_id)
            question_text.append(text1+' '+text2)
            r"""
            label_id.append(2)
            segment_id.append(question_segment)
            question_id.append(token_id)
            question_text.append(text1+' '+text2)
            """
with open(nezha_config_file,'r') as load_f:
    config = json.load(load_f)
config['embedding_size'] = config['hidden_size']
config['num_layers'] = config['num_hidden_layers']

from nezha import Bert
batch_size = 48
bertmodel = Bert(maxlen=max_seq_len,batch_size=batch_size,**config)
input_ids1 = keras.layers.Input(shape=(None,),dtype='int32',name="token_ids1")
input_ids2 = keras.layers.Input(shape=(None,),dtype='int32',name="segment_ids1")
output1 = bertmodel([input_ids1,input_ids2])
output1 = keras.layers.Lambda(lambda seq: seq[:,0,:])(output1)
output1 = keras.layers.Dropout(0.2)(output1)
#!!!这里取出0而不取出-1是因为最后一位可能有填充数值
#原先这里的dropout=0.5,现在的dropout=0.1
output1 = keras.layers.Dense(units=768,activation="tanh")(output1)
output1 = keras.layers.Dropout(0.2)(output1)
output1 = keras.layers.Dense(units=3,activation="softmax")(output1)

output2 = bertmodel([input_ids1,input_ids2])
output2 = keras.layers.Lambda(lambda seq: seq[:,0,:])(output2)
output2 = keras.layers.Dropout(0.2)(output2)
output2 = keras.layers.Dense(units=768,activation="tanh")(output2)
output2 = keras.layers.Dropout(0.2)(output2)
output2 = keras.layers.Dense(units=3,activation="softmax")(output2)
output = keras.layers.concatenate([output1,output2],axis=-1)
#这里不能使用CrossEntropy的原因在于label_id之中无法取出每一波的批次值，所以选择使用loss函数
#model = keras.Model(inputs=[input_ids1,input_ids2],outputs=[output1,output2])
model = keras.Model(inputs=[input_ids1,input_ids2],outputs=output)

from loader_weights_from_nezha import load_nezha_tf_weights_from_h5
load_nezha_tf_weights_from_h5(bertmodel,'/赛道三/tcdata/预训练天池地址匹配the_best_model.h5')

from sklearn.model_selection import KFold
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
result_point = []
test_point = [] 
#kf = KFold(n_splits=5,shuffle=True)
def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):
    print('create_learning_rate_scheduler')
    def lr_scheduler(epoch):
    #18年Facebook提出的gradual warmup刚开始训练的时候，模型的权重是随机
    #初始化的，此时若选择一个较大的学习率，可能带来模型的不确定(振荡),选择warm
    #up学习率的方式，可以使得开始训练的几个epoches或者一些steps学习率较小
    #在预热的小学习率下，模型可以慢慢趋于稳定，等模型相对稳定后再选择预先设置学习率
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    #Warmup不足之处在于从一个很小的学习率一下子变为比较大的学习率可能会导致训练误差突然增大
    #facebook提出了从最初的小学习率开始，每个step增大一点
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    #keras.callbacks.LearningRateScheduler(schedule,verbose=0)
    #schedule:接受epoch作为输入(整数，从0开始迭代)，然后返回一个学习率作为输出(浮点数)
    #verbose:0:安静,1:更新信息
    return learning_rate_scheduler
    
#split = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=42)
#split = StratifiedShuffleSplit(n_splits=5,test_size=0.1,random_state=42)
#分成5组，每组中的test=0.1也就是说循环之中使用的for循环每次从5组之中取出一组数据
#所以之前一直在用一组数据进行训练
class DataGenerator(keras.callbacks.Callback):
    def __init__(self,token_ids,segment_ids,label_ids,batch_size=48,max_seq_len=128):
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.batch_size = batch_size
        self.maxlen = max_seq_len
        self.totals = len(self.token_ids)
        self.best_score = 0
        self.train_token_ids = None
        self.test_token_ids = None
        self.train_segment_ids = None
        self.test_segment_ids = None
        self.train_label = None
        self.test_label = None
        #这个取出数值放在__init__函数之中，每次都是一样的数值，如果放在cycle
        #函数之中，每次取出来的数值都是不一样的
        
        self.SWA_START = 3
        split = StratifiedShuffleSplit(n_splits=5,test_size=0.1)
        for index1,index2 in split.split(self.token_ids,self.label_ids):
            self.train_token_ids = np.array(self.token_ids)[index1]
            self.test_token_ids = np.array(self.token_ids)[index2]
            self.train_segment_ids = np.array(self.segment_ids)[index1]
            self.test_segment_ids = np.array(self.segment_ids)[index2]
            self.train_label = np.array(self.label_ids)[index1]
            self.test_label = np.array(self.label_ids)[index2]
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.SWA_START))
    
    def __len__(self):
        return int(np.floor(len(self.token_ids)/self.batch_size))

    def sample(self,random=False):
        #sample拆分出训练集进行训练
        indices = list(range(len(self.train_token_ids)))
        np.random.shuffle(indices)
        for i in indices:
            yield self.train_token_ids[i],self.train_segment_ids[i],self.train_label[i]
    
    def __iter__(self,random=False):
        #__iter__拆分出对应的batch
        random = False
        batch_token_ids = []
        batch_segment_ids = []
        batch_label_ids = []
        currents = 0
        batch_data = []
        for token_ids,segment_ids,label_ids in self.sample(random):
            if len(token_ids) > self.maxlen*2:
            #!!!注意这里要为self.maxlen*2,因为单个长度为128，两个加起来不能超过256
                token_ids = token_ids[:self.maxlen]
                segment_ids = segment_ids[:self.maxlen]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_label_ids.append(label_ids)
            
            r"""
            k折交叉验证的时候就不加量了
            if label_ids == 2:
            #!!!标记部分
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_label_ids.append(label_ids)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_label_ids.append(label_ids)
            """
            #实验表明只加两波效果比较好
            currents = currents+1
            if len(batch_token_ids) == self.batch_size or currents == self.totals:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [np.array(batch_token_ids),np.array(batch_segment_ids)],np.array(batch_label_ids)
                #yield [np.array(batch_token_ids1),np.array(batch_segment_ids1)],np.array(batch_label_ids)
                batch_token_ids,batch_segment_ids,batch_label_ids = [],[],[]
                #这里必须要统一维度
                
    def cycle(self,random=True):
        #for index1,index2 in kf.split(self.token_ids1,self.token_ids1):
        flag = False
        split = StratifiedShuffleSplit(n_splits=5,test_size=0.1)
        #!!!这个split一定要定义在cycle之中，否则定义好了的random_state对应值一样
        for index1,index2 in split.split(self.token_ids,self.label_ids):
            self.train_token_ids = np.array(self.token_ids)[index1]
            self.test_token_ids = np.array(self.token_ids)[index2]
            self.train_segment_ids = np.array(self.segment_ids)[index1]
            self.test_segment_ids = np.array(self.segment_ids)[index2]
            self.train_label = np.array(self.label_ids)[index1]
            self.test_label = np.array(self.label_ids)[index2]
        while True:
            for d in self.__iter__(random):
                yield d
    
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        print('learning rate of current epoch is : {}'.format(lr))
    
    def on_epoch_end(self,epoch,logs):
        if epoch == self.SWA_START:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.SWA_START:
            for i, layer in enumerate(self.model.layers):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       (epoch - self.SWA_START) + self.model.get_weights()[i]) / (
                                              (epoch - self.SWA_START) + 1)
        else:
            pass
        if epoch >= 10:
            predict_label = []
            for i in tqdm(range(0,len(self.test_token_ids))):
                token_id = self.test_token_ids[i]
                segment_id = self.test_segment_ids[i]
                res = model.predict([np.array([token_id]),np.array([segment_id])])
                res = res[:,0:3]
                res = res.argmax(axis=-1)
                predict_label.append(res[0])
            from sklearn.metrics import f1_score
            val_f1 = f1_score(self.test_label,predict_label,labels=[0,1,2],average='macro')
            #result = f1_score(data1,data2,labels=[1,2,3],average='macro')
            final_score = val_f1
            if final_score > self.best_score:
                self.best_score = final_score
                print('self.best_score = ')
                print(self.best_score)
                model.save_weights('/赛道三/tcdata/天池地址匹配the_best_model.h5')
            
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('set stochastic weight average as final model parameters [FINISH].')

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

output_ids = model([input_ids1,input_ids2])
from tensorflow.keras.losses import kullback_leibler_divergence as kld
def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_pred1 = y_pred[:,0:3]
    y_pred2 = y_pred[:,3:6]
    ce_loss = (K.sparse_categorical_crossentropy(y_true,y_pred1)+K.sparse_categorical_crossentropy(y_true,y_pred2))/2
    kl_loss = kld(tf.nn.log_softmax(y_pred1,axis=-1), tf.nn.softmax(y_pred2,axis=-1)) + kld(tf.nn.log_softmax(y_pred1,axis=-1), tf.nn.softmax(y_pred2,axis=-1))/2
    return ce_loss+2*kl_loss

train_generator = DataGenerator(question_id,segment_id,label_id)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-5),
              #optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=crossentropy_with_rdrop,
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

model.fit(
    train_generator.cycle(),
    steps_per_epoch = len(train_generator),
    epochs = 15,
    callbacks = [train_generator]
)

#！！！！！！！！！
model.load_weights('/赛道三/tcdata/天池地址匹配the_best_model.h5')

f = open('/tcdata/test2_release.txt')
line = f.readline()
data = ['[']
while line:
    data.append(line)
    line = f.readline()
f.close()
data.append(']')

f = open('/赛道三/tcdata/test2_release.json','w')
for index in range(len(data)):
    if index == 0:
        f.write(data[index]+'\n')
        #[+\n
    elif index == len(data)-2:
        f.write(data[index])
    elif index == len(data)-1:
        #']'
        f.write(data[index])
    else:
        f.write(data[index][:-1]+','+'\n')
f.close()

with open('/赛道三/tcdata/test2_release.json','r') as load_f:
    test_data = json.load(load_f)

#for data in test_data:
for i in tqdm(range(0,len(test_data))):
    data = test_data[i]
    text1 = data['query']
    token1 = tokenizer.tokenize(text1)
    token1 = ["[CLS]"]+token1
    token_id1 = tokenizer.convert_tokens_to_ids(token1)
    if len(token_id1) > max_seq_len-1:
        token_id1 = token_id1[:max_seq_len-1]
    token_id1 = token_id1+tokenizer.convert_tokens_to_ids(["[SEP]"])
    segment_id1 = [0]*len(token_id1)
    for j in range(len(data['candidate'])):
        data1 = data['candidate'][j]
        text2 = data1['text']
        token2 = tokenizer.tokenize(text2)
        token_id2 = tokenizer.convert_tokens_to_ids(token2)
        if len(token_id2) > max_seq_len-1:
            token_id2 = token_id2[:max_seq_len-1]
        token_id2 = token_id2+tokenizer.convert_tokens_to_ids(["[SEP]"])
        segment_id2 = [1]*len(token_id2)
        current = [token_id1+token_id2]
        res = model.predict([np.array([token_id1+token_id2]),np.array([segment_id1+segment_id2])])
        res = res[:,0:3].argmax(axis=-1)
        if res[0] == 0:
            test_data[i]['candidate'][j]['label'] = '不匹配'
        elif res[0] == 1:
            test_data[i]['candidate'][j]['label'] = '部分匹配'
        else:
            test_data[i]['candidate'][j]['label'] = '完全匹配'

file_obj = open('/result.txt','w',encoding='utf-8')
for data in test_data:
    file_obj.write(json.dumps(data,ensure_ascii=False))
    file_obj.write('\n')
file_obj.close()
end = time.time()
run_time = end-start
print('共运行'+str(run_time//3600)+'小时'+str(run_time%3600//60)+'分钟')