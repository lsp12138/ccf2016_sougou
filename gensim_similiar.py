#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, sys
from gensim.models import word2vec
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
reload(sys)
sys.setdefaultencoding('utf-8')

class GMSG(object):

    def get_frequence_dict(self,name):

        all_frame = pd.read_csv('E:/Sogou_Project/new_information/new_word_std_%s.csv'%name)
        if name == 'age':
            frame_30_2 = all_frame[(all_frame['std']>0.02)]
        if name == 'education':
            frame_30_2 = all_frame[(all_frame['frequence']>300)&(all_frame['std']>0.007)]
        else:
            frame_30_2 = all_frame[(all_frame['frequence']>100)&(all_frame['std']>0.007)]
        feature_word_dict = {}
        df = frame_30_2.dropna().sort(['frequence'], ascending=False).reset_index()[['word', 'frequence']]
        for i in range(len(df)):
            feature_word_dict[df['word'][i]] = i
        return feature_word_dict

    def get_split_word_list(self, name):
        user_df = pd.read_csv('E:/Sogou_Project/new_information/train_new_all_information.csv',na_values='0')
        age_df = user_df[['content',name]]
        age_df = age_df.dropna()
        words = age_df[['content']].values
        # 将搜索词串放入列表
        word_list = []
        for i in range(len(words)):
            word_list.append(words[i][0])
        # 搜索词串分割
        split_word_list = []
        for words in word_list:
            cur_split = words.split()
            split_word_list.append(cur_split)
        return split_word_list,age_df

    def to_numpy(self, feature_word_dict, split_word_list):
        model = word2vec.Word2Vec.load_word2vec_format(r"E:\Sogou_Project\sogou_vectors1.bin", binary=True)

        #model = word2vec.Word2Vec.load(r"E:\Sogou_Project\sogou_vectors1.bin")
        feature_array = [[0.0 for i in range(len(feature_word_dict))] for j in range(len(split_word_list))]
        for i in range(len(split_word_list)):
            print 'dealing with %d line'%i
            for word in split_word_list[i]:
                if word in feature_word_dict:
                    feature_array[i][feature_word_dict[word]] = 1
                else:
                    if u'%s'%word in model:
                        similar_word_list =  model.most_similar(u'%s'%word, topn=50)  # 50个最相关的词
                        for item in similar_word_list:
                            if item[0].encode('utf-8') in feature_word_dict and item[1] > 0.6:
                                feature_array[i][feature_word_dict[item[0].encode('utf-8')]] = 1
                                #break
        a = np.array(feature_array)
        return a

    def split_np(self, all_content_np, name,age_df):
        np.save('E:/Sogou_Project/new_information/gensim_similar/np_%s' % name,all_content_np)
        np.save('E:/Sogou_Project/new_information/gensim_similar/np_%s_xtrain' % name, all_content_np[:17000])
        np.save('E:/Sogou_Project/new_information/gensim_similar/np_%s_xtest' % name, all_content_np[17000:])
        array_tag = age_df[[name]].values
        np.save('E:/Sogou_Project/new_information/gensim_similar/np_%s_ytrain' % name, array_tag[:17000])
        np.save('E:/Sogou_Project/new_information/gensim_similar/np_%s_ytest' % name, array_tag[17000:])



    def train_get_acc(self):
        x_train = np.load('E:/Sogou_Project/new_information/gensim_similar/np_%s_xtrain.npy' % name)
        x_test = np.load('E:/Sogou_Project/new_information/gensim_similar/np_%s_xtest.npy' % name)
        y_trains = np.load('E:/Sogou_Project/new_information/gensim_similar/np_%s_ytrain.npy' % name)
        y_tests = np.load('E:/Sogou_Project/new_information/gensim_similar/np_%s_ytest.npy' % name)
        y_test = []
        y_train = []

        for i in range(len(y_trains)):
            y_train.append(y_trains[i][0])
        for i in range(len(y_tests)):
            y_test.append(y_tests[i][0])
        print len(x_train), len(y_train), len(x_test), len(y_test)

        mod = RandomForestClassifier(n_estimators=300, max_features='auto')
        mod.fit(x_train, y_train)
        acc = accuracy_score(y_test, mod.predict(x_test))
        # model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, random_state=0, max_features='auto')
        # model.fit(x_train, y_train)
        # y_predict = model.predict(x_test)
        # y_predict = np.around(y_predict)
        # acc = accuracy_score(y_test, y_predict)
        pp = '%s Gensim_similar Random Forest model acc is %s' % (name, acc)
        print pp
        with open('E:/Sogou_Project/new_information/gensim_similar/results1.txt','a') as f:
            f.write(pp)
            f.write('\n')







if __name__ == '__main__':

    name_lists = ['age','sex','education']
    #name_lists = ['sex']
    gmsg = GMSG()
    print "processing..."
    for name in name_lists:
        feature_word_dict = gmsg.get_frequence_dict(name)
        split_word_list,age_df = gmsg.get_split_word_list(name)
        all_content_np = gmsg.to_numpy(feature_word_dict,split_word_list)
        gmsg.split_np(all_content_np,name,age_df)
        gmsg.train_get_acc()


