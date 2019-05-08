import os
import re
import sys
import commands
import multiprocessing

import pickle
import numpy as np

from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from gensim.models import KeyedVectors

max_features=200
max_document_length=500
min_opcode_count=2

webshell_dir="../data/webshell/webshell/PHP/"
whitefile_dir="../data/webshell/normal/php/"

white_count=0
black_count=0

php_bin="/usr/bin/php7.2"
word2vec_bin="word2vec.bin"
bigram_word2vec_bin="bigram_word2vec.bin"

data_pkl_file="data-webshell-opcode.pkl"
label_pkl_file="label-webshell-opcode.pkl"

wv_data_pkl_file="wv-data-webshell-opcode.pkl"
bigram_wv_data_pkl_file="bigram-wv-data-webshell-opcode.pkl"


def load_files_opcode_re(dir):
    global min_opcode_count

    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        for filename in filelist:
            if filename.endswith('.php') :
                fulepath = os.path.join(path, filename)
                print "Load %s opcode" % fulepath
                t = load_file_opcode(fulepath)
                if len(t) > min_opcode_count:
                    files_list.append(t)
                else:
                    print "Load %s opcode failed" % fulepath

    return files_list

def load_file_opcode(file_path):
    global php_bin

    t=""
    cmd=php_bin+" -dvld.active=1 -dvld.execute=0 "+file_path
    status,output=commands.getstatusoutput(cmd)
    t=output
    tokens=re.findall(r'\s(\b[A-Z_]+\b)\s',output)
    t=" ".join(tokens)
    print "opcode count %d" % len(t)
    return t

def load_data_pkl_file():
    global white_count
    global black_count
    global webshell_dir
    global whitefile_dir
    
    x = []
    y = []

    if os.path.exists(data_pkl_file) and os.path.exists(label_pkl_file):
        f = open(data_pkl_file, 'rb')
        x = pickle.load(f)
        f.close()
        f = open(label_pkl_file, 'rb')
        y = pickle.load(f)
        f.close()

    else:
        webshell_files_list = load_files_opcode_re(webshell_dir)
        y1=[1]*len(webshell_files_list)
        black_count=len(webshell_files_list)

        wp_files_list =load_files_opcode_re(whitefile_dir)
        y2=[0]*len(wp_files_list)
        white_count=len(wp_files_list)

        x=webshell_files_list+wp_files_list
        y=y1+y2

        f = open(data_pkl_file, 'wb')
        pickle.dump(x, f)
        f.close()
        f = open(label_pkl_file, 'wb')
        pickle.dump(y, f)
        f.close()    
    return x,y

def get_feature_by_opcode_2gram():
    global max_features

    with open('metrics.txt', 'a') as f:
        f.write("Get feature by opcode and word-bag 2-gram: \n")
        f.close()
    print "max_features=%d webshell_dir=%s whitefile_dir=%s" % (max_features,webshell_dir,whitefile_dir)

    x,y = load_data_pkl_file()

    CV = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
    x=CV.fit_transform(x).toarray()

    return x,y

def get_feature_by_opcode_sequences():
    global max_document_length
 
    with open('metrics.txt', 'a') as f:
        f.write("Get feature by opcode sequences: \n")
        f.close()

    x,y = load_data_pkl_file()

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x=np.array(list(x))
    print np.shape(x)
    print np.shape(y)
    return x,y

def getVecsByWord2Vec(model, corpus, size):
    global max_document_length

    all_vectors = []
    embeddingDim = model.vector_size
    embeddingUnknown = [0. for i in range(embeddingDim)]
    #逐句
    for text in corpus:
        this_vector = []
        #切除掉最大文档长度后的词
        text = text[:max_document_length]
        #逐词
        for i,word in enumerate(text):
            if word in model.wv.vocab:        
                this_vector.append(model[word])
            else:
                this_vector.append(embeddingUnknown)
        dim = np.shape(this_vector)
        #不足长度的填充至一直长度
        if dim[0] < max_document_length:    
            pad_length = max_document_length-i-1
            for n in range(0,pad_length):
                this_vector.append(embeddingUnknown)    
        all_vectors.append(this_vector)

    x = np.array(all_vectors)

    return x

def get_feature_by_opcode_word2vec():
    global max_document_length

    with open('metrics.txt', 'a') as f:
        f.write("Get feature by opcode and word2vec: \n")
        f.close()

    x = []
    y = []

    if os.path.exists(wv_data_pkl_file) and os.path.exists(label_pkl_file):
        f = open(wv_data_pkl_file, 'rb')
        x = pickle.load(f)
        f.close()
        f = open(label_pkl_file, 'rb')
        y = pickle.load(f)
        f.close()
    else:
        x, y = load_data_pkl_file()

        cores=multiprocessing.cpu_count()
        #训练词向量
        if os.path.exists(word2vec_bin):
            print "Find cache file %s" % word2vec_bin
            model=gensim.models.Word2Vec.load(word2vec_bin)
        else:
            model=gensim.models.Word2Vec(size=max_features, window=5, min_count=5, iter=10, workers=cores)
            model.build_vocab(x)
            model.train(x, total_examples=model.corpus_count, epochs=model.iter)
            model.save(word2vec_bin)

        x = getVecsByWord2Vec(model, x, max_features)

        f = open(wv_data_pkl_file, 'wb')
        pickle.dump(x, f)
        f.close()

    return x,y

def get_feature_by_opcode_bigram_word2vec():
    global max_document_length
    global bigram_word2vec_bin

    with open('metrics.txt', 'a') as f:
        f.write("Get feature by opcode and bigram word2vec: \n")
        f.close()

    x = []
    y = []
    
    if os.path.exists(bigram_wv_data_pkl_file) and os.path.exists(label_pkl_file):
        f = open(bigram_wv_data_pkl_file, 'rb')
        x = pickle.load(f)
        f.close()
        f = open(label_pkl_file, 'rb')
        y = pickle.load(f)
        f.close()
    else:
        x,y = load_data_pkl_file()
        
        CV = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
        # 2-gram分词
        analyze = CV.build_analyzer()
        courps = []
        for text in x:
            text = analyze(text)
            text = str(text).replace('u\'','\'')
            courps.append(str(text))
        x = courps
        
        cores=multiprocessing.cpu_count()

        if os.path.exists(bigram_word2vec_bin):
            print "Find cache file %s" % bigram_word2vec_bin
            model=gensim.models.Word2Vec.load(bigram_word2vec_bin)
        else:
            model=gensim.models.Word2Vec(size=max_features, window=5, min_count=5, iter=10, workers=cores)
            model.build_vocab(x)
            model.train(x, total_examples=model.corpus_count, epochs=model.iter)
            model.save(bigram_word2vec_bin)    

        x = getVecsByWord2Vec(model, x, max_features)

        f = open(bigram_wv_data_pkl_file, 'wb')
        pickle.dump(x, f)
        f.close()

    return x,y

def do_metrics(y_test,y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion = metrics.confusion_matrix(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test,y_pred)
    
    print "metrics.accuracy_score:"+str(accuracy)
    print "metrics.confusion_matrix:"+str(confusion)
    print "metrics.precision_score:"+str(precision)
    print "metrics.recall_score:"+str(recall)
    print "metrics.f1_score:"+str(f1_score)
    
    with open('metrics.txt', 'a') as f:
        f.write("accuracy:"+str(accuracy) + '\n' +"confusion:\n"+ str(confusion) + '\n' +"precision:" +str(precision) + '\n' +"recall"+ str(recall) + '\n' +"f1_score"+ str(f1_score )+ '\n'+'\n')
        f.close()

def do_cnn_1d(x,y):
    global max_document_length

    print "CNN"

    with open('metrics.txt', 'a') as f:
        f.write("CNN: \n")
        f.close()

    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.4, random_state=0)
    y_test=testY

    # Padding to the max document length
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length], name='input')
    # Word  embedding
    network = tflearn.embedding(network, input_dim=100000, output_dim=max_features)
    # Convolutional kernel
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    # Training
    model.fit(trainX, trainY,
                  n_epoch=5, shuffle=True, validation_set=0.1,
                  show_metric=True, batch_size=100,run_id="webshell")

    # Do metrics 
    y_predict_list=model.predict(testX)
    y_predict=[]
    for i in y_predict_list:
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    do_metrics(y_test, y_predict)

def do_cnn_word2vec(x,y):
    global max_document_length
    print "CNN"
    with open('metrics.txt', 'a') as f:
        f.write("CNN: \n")
        f.close()
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.4, random_state=0)
    y_test=testY

    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length,max_features], name='input')
    #不再需要将特征向量化
    # network = tflearn.embedding(network, input_dim=100000, output_dim=max_features)
    branch1 = conv_1d(network, 200, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 200, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 200, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)

    model.fit(trainX, trainY,
                  n_epoch=5, shuffle=True, validation_set=0.1,
                  show_metric=True, batch_size=100,run_id="webshell")

    y_predict_list=model.predict(testX)
    y_predict=[]
    for i in y_predict_list:
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    do_metrics(y_test, y_predict)

def get_multi_channel_feature():
    y=[]
    x1, y1 = get_feature_by_opcode_bigram_word2vec()
    x2, y2 = get_feature_by_opcode_word2vec()
    if y1 == y2:
        y = y1
    else:
        print "y error"
    return x1,x2,y

def do_dctfcnn(x1,x2,y):
    global max_document_length
    global max_features
    print "CNN"
    with open('metrics.txt', 'a') as f:
        f.write("dcCNN: \n")
        f.close()

    # 划分训练测试集
    trainX1, testX1, trainX2, testX2, trainY, testY = train_test_split(x1,x2,y, test_size=0.4, random_state=0)
    y_test=testY

    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    net1 = input_data(shape=[None,max_document_length,max_features], name='input1')
    net2 = input_data(shape=[None,max_document_length,max_features], name='input2')

    branch11 = conv_1d(net1, 200, 3, padding='valid', activation='relu', regularizer="L2")
    branch12 = conv_1d(net1, 200, 4, padding='valid', activation='relu', regularizer="L2")
    branch13 = conv_1d(net1, 200, 5, padding='valid', activation='relu', regularizer="L2")
    net1 = merge([branch11, branch12, branch13], mode='concat', axis=1)
    net1 = tf.expand_dims(net1, 2)
    net1 = global_max_pool(net1)
    net1 = dropout(net1, 0.8)

    branch21 = conv_1d(net2, 200, 3, padding='valid', activation='relu', regularizer="L2")
    branch22 = conv_1d(net2, 200, 4, padding='valid', activation='relu', regularizer="L2")
    branch23 = conv_1d(net2, 200, 5, padding='valid', activation='relu', regularizer="L2")
    net2 = merge([branch21, branch22, branch23], mode='concat', axis=1)
    net2 = tf.expand_dims(net2, 2)
    net2 = global_max_pool(net2)
    net2 = dropout(net2, 0.8)

    network = merge([net1, net2], mode='concat', axis=1)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
        
    # Training
    model.fit([trainX1,trainX2],trainY,
                  n_epoch=5, shuffle=True, validation_set=([testX1,testX2],testY),
                  show_metric=True, batch_size=100,run_id="webshell")

    y_predict_list=model.predict([testX1,testX2])
    y_predict=[]
    for i in y_predict_list:
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    do_metrics(y_test, y_predict)

if __name__ == '__main__':

    max_features=200
    max_document_length=500
    print "max_features=%d max_document_length=%d" % (max_features,max_document_length)

    # x,y = get_feature_by_opcode_2gram()
    # do_cnn(x,y)

    # x,y = get_feature_by_opcode_sequences()
    # do_cnn(x,y)

    # x,y = get_feature_by_opcode_word2vec()
    # do_cnn_word2vec(x,y)

    # x,y = get_feature_by_opcode_bigram_word2vec()
    # do_cnn_word2vec(x,y)

    x1,x2,y = get_multi_channel_feature()
    do_dctfcnn(x1, x2, y)