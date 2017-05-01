# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
from compiler.ast import flatten
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors

#去除NA值：prop：用户属性，如age,gender,education返回一个DataFrame
def filter_NA(prop):
	all_info_df = pd.read_csv(FILE_ROOT + "train_all_info.csv", sep = ',', na_values = 0)
	prop_df = all_info_df[[prop, 'content']]
	prop_df = prop_df.dropna()
	prop_df.to_csv(FILE_ROOT + prop + "/info.csv", sep = ',' , index = None)
	print(prop_df.head())
	return prop_df

#获取当前表格的分词后的搜索词串列表
def get_words_list(word_df):
	#获取训练集的搜索词
	words = word_df[['content']].values
	#将搜索词串放入列表
	words_list = []
	for i in range(len(words)):
		words_list.append(words[i][0])
	return words_list

#将搜索词串分开返回列表
def get_split_list(words_list):
	#搜索词串分割
	split_list = []
	for words in words_list:
		cur_split = words.split()
		split_list.append(cur_split)
	return split_list

#所有用户搜索词合并，返回一个list
def get_all_words(split_list):
	words = flatten(split_list)
	return words

#生成词频
def get_word_frequency(prop, start, end):
	prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
	words = get_all_words(get_split_list(get_words_list(prop_df)))
	#print(words[:100])
	word_frequency_dict = Counter(words)
	word_list = []
	frequency_list = []
	for key in word_frequency_dict.keys():
		word_list.append(key)
		frequency_list.append(word_frequency_dict[key])
	data = {'word': word_list, 'frequency': frequency_list}
	word_frequency_df = pd.DataFrame(data, index = None)
	word_frequency_df.to_csv(FILE_ROOT + prop + "/word_frequency/word_frequency.csv", sep = ',', index = None)
	for i in range (start, end):
		cur_df = word_frequency_df[word_frequency_df['frequency']>=i*10]
		cur_df.to_csv(FILE_ROOT + prop + "/word_frequency/word_frequency_" + str(i*10) + ".csv", sep = ',', index = None)
	return True

#生成每个属性不同类别的词频
def get_prop_word_frequency(prop):
	if prop == 'gender':
		prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
		prop_1_df = prop_df[prop_df[prop] == 1]
		prop_2_df = prop_df[prop_df[prop] == 2]
		words = get_all_words(get_split_list(get_words_list(prop_df)))
		words_1 = get_all_words(get_split_list(get_words_list(prop_1_df)))
		words_2 = get_all_words(get_split_list(get_words_list(prop_2_df)))
		word_frequency_dict = Counter(words)
		word_1_frequency_dict = Counter(words_1)
		word_2_frequency_dict = Counter(words_2)
		prop_frequency_df =pd.DataFrame(columns = ['1', '2','sum', 'ratio_1', 'ratio_2', 'ratio_dif'], index=words)
		#prop_frequency_df[['word']] = words
		print "dicting..."
		for word in word_frequency_dict:
			if word in word_1_frequency_dict:
				print word + "yes 1"
				prop_frequency_df['1'][word] = word_1_frequency_dict[word]
			if word in word_2_frequency_dict:
				print  word + "yes 2"
				prop_frequency_df['2'][word] = word_2_frequency_dict[word]

		print "counting..."
		prop_frequency_df.fillna(0)
		prop_frequency_df[['sum']] = prop_frequency_df[['1']] + prop_frequency_df[['2']]
		prop_frequency_df[['ratio_1']] = prop_frequency_df[['1']] / float(prop_frequency_df[['sum']])
		prop_frequency_df[['ratio_2']] = prop_frequency_df[['2']] / float(prop_frequency_df[['sum']])
		prop_frequency_df[['ratio_dif']] = abs(prop_frequency_df[['ratio_1']] - prop_frequency_df[['ratio_2']])

		print(prop_frequency_df.head())
		prop_frequency_df.to_csv(FILE_ROOT + prop +"/prop_frequency.csv", sep = ',')

	return True

#根据词频生成特征向量
def get_wf_feature(prop, start, end):
	prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
	split_word_list = get_split_list(get_words_list(prop_df))
	print "user length: " + str(len(split_word_list))
	for i in range(start, end):
		feature_word_dict = {}
		df = pd.read_csv(FILE_ROOT + prop + "/word_frequency/word_frequency_" + str(i*10) + ".csv", sep = ',')
		df = df.sort(['frequency'], ascending= False)
		feature_word_dict = {}
		#遍历写入字典，key：value = 特征词：位置
		print "df length: " + str(len(df))

		for j in range(len(df)):
			feature_word_dict[df['word'][j]] = j 
		print "dict length: " + str(len(feature_word_dict))
    	#搜索词列表与字典比对形成 特征向量矩阵,向量维度为特征词个数，向量个数为用户个数
		feature_array = [[0 for j in range(len(feature_word_dict))] for k in range(len(split_word_list))]
		for j in range(len(split_word_list)):
   			for word in split_word_list[j]:
   			#对于当前用户的每一个搜索词，如果字典中有，则特征向量对应位置赋值1
   				if word in feature_word_dict:
   					feature_array[j][feature_word_dict[word]] += 1
		a = np.array(feature_array)
		np.save(FILE_ROOT + prop + "/word_frequency/feature_" + str(i*10), a)
		print(a[:100])
	return True

#根据词用户频方差生成特征向量：
def get_fre_std_feature(prop, std_start = 1, std_end = 2, fre_start = 1, fre_end = 5):
	prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
	split_word_list = get_split_list(get_words_list(prop_df))
	print len(split_word_list)
	for i in range(fre_start, fre_end):
		for j in range(std_start, std_end):
			df = pd.read_csv(FILE_ROOT + prop + "/std/fre_" + str(i*10) + "_std_" + str(j*0.1) + ".csv", sep = ',')
			df = df.sort(['sum'], ascending = False)
			print "df len:" + str(len(df))
			feature_word_dict = {}
			for k in range(len(df)):
				feature_word_dict[df['word'][k]] = k

			print "fre: " + str(i*10) + "  std: " + str(j*0.1) + "  feature_length: " + str(len(feature_word_dict))
			feature_array = [[0 for l in range(len(feature_word_dict))] for m in range(len(split_word_list))]
			print "features: " + str(len(feature_array)) + "lists: " + str(len(split_word_list))
			for k in range(len(split_word_list)):
   				for word in split_word_list[k]:
   				#对于当前用户的每一个搜索词，如果字典中有，则特征向量对应位置赋值1
   					if word in feature_word_dict:
   						feature_array[k][feature_word_dict[word]] += 1
			a = np.array(feature_array)
			np.save(FILE_ROOT + prop + "/std/feature_fre_" + str(i*10) + "_std_" + str(j*0.1), a)

#根据词使用人数生成特征向量
def get_word_user_feature(prop, fre_start = 1, fre_end = 5):
	prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
	split_word_list = get_split_list(get_words_list(prop_df))
	print len(split_word_list)
	for i in range(fre_start, fre_end):
		cur_num = i*10
		df = pd.read_csv(FILE_ROOT + prop + "/word_user/word_user_" + str(cur_num) + ".csv", sep = ',')
		
		df = df.sort(['sum'], ascending = False)

		print "df len:" + str(len(df))
		#
		feature_word_dict = {}

		for k in range(len(df)):
			feature_word_dict[df['word'][k]] = k
		print "word_user: " + str(cur_num) + "  feature_length: " + str(len(feature_word_dict))

		feature_array = [[0 for l in range(len(feature_word_dict))] for m in range(len(split_word_list))]

		for k in range(len(split_word_list)):
   			for word in split_word_list[k]:
   			#对于当前用户的每一个搜索词，如果字典中有，则特征向量对应位置赋值1
   				if word in feature_word_dict:
   					feature_array[k][feature_word_dict[word]] = 1
		a = np.array(feature_array)
		np.save(FILE_ROOT + prop + "/word_user/feature_word_user_" + str(cur_num), a)



def get_word_user_test_feature(prop):

	test_df = pd.read_csv(FILE_ROOT + "/test_word_noun.csv", header=None, names=['content'])
	print "test length: " + str(len(test_df))
	#prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
	split_word_list = get_split_list(get_words_list(test_df))
	print len(split_word_list)

	for i in range(0, 1):
		cur_num = 10 + i*10
		df = pd.read_csv(FILE_ROOT + prop + "/word_user/word_user_" + str(cur_num) + ".csv", sep = ',')
		df = df.sort(['sum'], ascending = False)
		print "word_user_df len:" + str(len(df))

		feature_word_dict = {}

		for k in range(len(df)):
			feature_word_dict[df['word'][k]] = k
		print "word_user: " + str(cur_num) + "  feature_length: " + str(len(feature_word_dict))

		feature_array = [[0 for l in range(len(feature_word_dict))] for m in range(len(split_word_list))]

		for k in range(len(split_word_list)):
   			for word in split_word_list[k]:
   			#对于当前用户的每一个搜索词，如果字典中有，则特征向量对应位置赋值1
   				if word in feature_word_dict:
   					feature_array[k][feature_word_dict[word]] = 1
   					
		a = np.array(feature_array)

		np.save(FILE_ROOT + prop + "/word_user/test_feature_word_user_" + str(cur_num), a)
	return



#统计每个词的用户使用人数
def get_word_user_num(prop, num):
	prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
	word_frequency_df = pd.read_csv(FILE_ROOT + prop + "/word_frequency/word_frequency.csv", sep = ',')
	prop_word = word_frequency_df['word'].values
	for i in range(num):

		cur_df = prop_df[prop_df[prop] == i+1]
		word_list = get_split_list(get_words_list(cur_df))

		word_dict = {}
		for j in range(len(prop_word)):
			word_dict[prop_word[j]] = 0

		print " dict len: ", len(word_dict)
		print "user " + str(i+1) + " len: ", len(word_list)

		for j in range(len(word_list)):
			word_hash = {}
			for word in word_list[j]:
				if word in word_dict and word not in word_hash:
					word_dict[word] += 1
					word_hash[word] = 0
		with open(FILE_ROOT + prop + "/std/word_user_cnt_" + str(i+1) + ".csv", 'a') as f :
			for j in word_dict.keys():
				f.write(j + ", " + str(word_dict[j]) + '\n')			


#计算单词使用人数占总人数比,方差
def get_word_user_ratio(prop, num):
	prop_ratio_df = pd.DataFrame(columns = ['word', 'sum', 'sum_ratio'])
	std_df = pd.DataFrame()
	prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
	all_user_num = len(prop_df)
	for i in range(num):
		cur_df = pd.read_csv(FILE_ROOT + prop + "/std/word_user_cnt_" + str(i+1) + ".csv", sep = ',', names = ['word', 'cnt'])
		prop_ratio_df[[str(i+1)]] = cur_df[['cnt']]
		prop_ratio_df[['word']] = cur_df[['word']]
	prop_ratio_df['sum'] = 0
	#prop_ratio_df['sum_ratio'] = 0
	for i in range(num):
		prop_ratio_df['sum'] += prop_ratio_df[str(i+1)]

	prop_ratio_df['sum_ratio'] = prop_ratio_df['sum'] / float(all_user_num)

	for i in range(num):
		prop_ratio_df['ratio_' + str(i+1)] = prop_ratio_df[str(i+1)] / prop_ratio_df['sum']
		std_df['ratio_' + str(i+1)] = prop_ratio_df[str(i+1)] / prop_ratio_df['sum']
	std_df = std_df.std(axis = 1)
	prop_ratio_df['std'] = std_df

	prop_ratio_df.to_csv(FILE_ROOT + prop + "/std/word_user_cnt_ratio.csv", sep = ',', index = None)


	print(prop_ratio_df.head())
	print(std_df.head())

#查看单词使用人数，方差情况：
def get_word_user_std_csv(prop, std_start = 1, std_end = 2, fre_start = 1, fre_end = 5):
	prop_ratio_df = pd.read_csv(FILE_ROOT + prop + "/std/word_user_cnt_ratio.csv", sep = ',')
	#print(prop_ratio_df.head())
	#top_df = prop_ratio_df[prop_ratio_df['sum'] > 500]
	#print(len(top_df))
	#top_df = top_df[top_df['std']> 0.3]
	#print(len(top_df))
	for i in range(fre_start, fre_end):
		cur_df = prop_ratio_df[prop_ratio_df['sum'] > i*10]
		for j in range(std_start, std_end):
			cur_df = cur_df[cur_df['std'] > j*0.1]
			cur_df.to_csv(FILE_ROOT + prop + "/std/fre_" + str(i*10) + "_std_" + str(j*0.1) + ".csv", sep = ',', index = None)
	
#获得单次使用人数分级表
def get_word_user_csv(prop, fre_start = 1, fre_end = 5):
	prop_ratio_df = pd.read_csv(FILE_ROOT + prop + "/std/word_user_cnt_ratio.csv", sep = ',')
	print(prop_ratio_df.head())
	for i in range(fre_start, fre_end):
		cur_num = i*10
		cur_df = prop_ratio_df[prop_ratio_df['sum'] > cur_num]
		print(prop, len(cur_df))
		cur_df.to_csv(FILE_ROOT + prop + "/word_user/word_user_" + str(cur_num) + ".csv", sep = ',', index = None)



#np转list
def np_to_list(np_array):
	cur_list = []
	for i in range(len(np_array)):
		cur_list.append(np_array[i][0])
	return cur_list

#保存标签用于训练
def save_label(prop):
	prop_df = pd.read_csv(FILE_ROOT + prop + "/info.csv", sep = ',')
	label = prop_df[prop].values
	np.save(FILE_ROOT + prop + "/label", label)
	print(label[:100])
	return True

#开始训练
def train_wf_rf_model(prop, frequency):
	feature_array = np.load(FILE_ROOT + prop + "/word_frequency/feature_" + str(frequency) + ".npy")
	label_array = np.load(FILE_ROOT + prop + "/label.npy")
	x_train = feature_array[:int(0.8*len(feature_array))]
	x_test = feature_array[int(0.8*len(feature_array)):]
	y_train = label_array[:int(0.8*len(label_array))]
	y_test = label_array[int(0.8*len(label_array)):]

	print len(x_train), len(x_test), len(y_train), len(y_test)
	print "training..."
	#随机森林
	model = RandomForestClassifier(n_estimators=200, max_features='auto')
	model.fit(x_train,y_train)
	y_predict = model.predict(x_test)
	acc = accuracy_score(y_test, y_predict)
	pp = '%s_%d_Random Forest model acc is %s' %(prop, frequency, acc) + "\n"
	print '%s_%d_Random Forest model acc is %s' %(prop, frequency, acc)
	return pp

#开始训练wf_std模型
def train_wf_std_rf_model(prop, fre, std):
	feature_array = np.load(FILE_ROOT + prop + "/std/feature_fre_" + str(fre) + "_std_" + str(std) + ".npy")
	print feature_array.shape[1]
	if feature_array.shape[1] < 1:
		return "few features." + '\n'
	label_array = np.load(FILE_ROOT + prop + "/label.npy")
	x_train = feature_array[:int(0.8*len(feature_array))]
	x_test = feature_array[int(0.8*len(feature_array)):]
	y_train = label_array[:int(0.8*len(label_array))]
	y_test = label_array[int(0.8*len(label_array)):]

	print len(x_train), len(x_test), len(y_train), len(y_test)
	print "training..."
	#随机森林
	model = RandomForestClassifier(n_estimators=200, max_features='auto')
	model.fit(x_train,y_train)
	y_predict = model.predict(x_test)
	acc = accuracy_score(y_test, y_predict)
	pp = '%s_fre_%s_std_%s_Random Forest model acc is %s' %(prop, str(fre), str(std), acc) + "\n"
	print pp
	return pp

#开始训练word_user模型
def train_word_user_rf_model(prop, num):
	feature_array = np.load(FILE_ROOT + prop + "/word_user/feature_word_user_" + str(num) + ".npy")
	print feature_array.shape[1]
	if feature_array.shape[1] < 1:
		return "few features." + '\n'
	label_array = np.load(FILE_ROOT + prop + "/label.npy")
	x_train = feature_array[:int(0.8*len(feature_array))]
	x_test = feature_array[int(0.8*len(feature_array)):]
	y_train = label_array[:int(0.8*len(label_array))]
	y_test = label_array[int(0.8*len(label_array)):]

	print len(x_train), len(x_test), len(y_train), len(y_test)
	print "training..."
	#随机森林
	model = RandomForestClassifier(n_estimators=200, max_features='auto')
	model.fit(x_train,y_train)
	y_predict = model.predict(x_test)
	acc = accuracy_score(y_test, y_predict)
	pp = '%s_word_user_%s_Random Forest model acc is %s' %(prop, str(num), acc) + "\n"
	print pp
	return pp


#开始训练gbdt
def train_wf_gbdt_model(prop, frequency):
	feature_array = np.load(FILE_ROOT + prop + "/feature_" + str(frequency) + ".npy")
	label_array = np.load(FILE_ROOT + prop + "/label.npy")
	x_train = feature_array[:int(0.8*len(feature_array))]
	x_test = feature_array[int(0.8*len(feature_array)):]
	y_train = label_array[:int(0.8*len(label_array))]
	y_test = label_array[int(0.8*len(label_array)):]

	print len(x_train), len(x_test), len(y_train), len(y_test)
	print "training..."
	#gbdt
	model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, random_state=0, max_features='auto')
	model.fit(x_train, y_train)
	y_predict = model.predict(x_test)
	y_predict = np.around(y_predict)
	print(y_predict[:100])
	print(y_test[:100])
	acc = accuracy_score(y_test, y_predict)
	pp = '%s_%d_gbdt model acc is %s' %(prop, frequency, acc) + "\n"
	print pp
	return pp

def train():
	result_file = open(FILE_ROOT +"wf_gbdt_result.txt", 'w')
	result = []
	for i in range(10):
		result.append(train_wf_gbdt_model(GENDER, 1000 - i*100))
	for i in range(10):
		result.append(train_wf_gbdt_model(AGE, 1000 - i*100))
	for i in range(10):
		result.append(train_wf_gbdt_model(EDUCATION, 1000 - i*100))
	result_file.writelines(result)
	result_file.close()




def train_word_user_knn_model(prop, num):
	feature_array = np.load(FILE_ROOT + prop + "/word_user/feature_word_user_" + str(num) + ".npy")
	print feature_array.shape[1]
	if feature_array.shape[1] < 1:
		return "few features." + '\n'
	label_array = np.load(FILE_ROOT + prop + "/label.npy")
	x_train = feature_array[:int(0.8*len(feature_array))]
	x_test = feature_array[int(0.8*len(feature_array)):]
	y_train = label_array[:int(0.8*len(label_array))]
	y_test = label_array[int(0.8*len(label_array)):]

	print len(x_train), len(x_test), len(y_train), len(y_test)
	print "training..."
	#随机森林
	model = neighbors.KNeighborsClassifier() #取得knn分类器
	model.fit(x_train,y_train)
	y_predict = model.predict(x_test)
	acc = accuracy_score(y_test, y_predict)
	pp = '%s_word_user_%s_Random Forest model acc is %s' %(prop, str(num), acc) + "\n"
	print pp
	return pp

def train_wf_knn_model(prop, frequency):
	feature_array = np.load(FILE_ROOT + prop + "/feature_" + str(frequency) + ".npy")
	label_array = np.load(FILE_ROOT + prop + "/label.npy")
	x_train = feature_array[:int(0.8*len(feature_array))]
	x_test = feature_array[int(0.8*len(feature_array)):]
	y_train = label_array[:int(0.8*len(label_array))]
	y_test = label_array[int(0.8*len(label_array)):]

	print len(x_train), len(x_test), len(y_train), len(y_test)
	print "training..."
	#gbdt
	model = neighbors.KNeighborsClassifier() #取得knn分类器
	model.fit(x_train, y_train)
	y_predict = model.predict(x_test)
	y_predict = np.around(y_predict)
	print(y_predict[:100])
	print(y_test[:100])
	acc = accuracy_score(y_test, y_predict)
	pp = '%s_%d_knn model acc is %s' %(prop, frequency, acc) + "\n"
	print pp
	return pp

#nn模型输入预处理
def process_label(label_array, prop):
	a =[]
	if prop == GENDER:
		for i in range (len(label_array)):
			if label_array[i] == 1:
				a.append([1,0])
			else:
				a.append([0,1])
	else:
		for i in range (len(label_array)):
			if label_array[i] == 1:
				a.append([1,0,0,0,0,0])
			if label_array[i] == 2:
				a.append([0,1,0,0,0,0])
			if label_array[i] == 3:
				a.append([0,0,1,0,0,0])
			if label_array[i] == 4:
				a.append([0,0,0,1,0,0])
			if label_array[i] == 5:
				a.append([0,0,0,0,1,0])
			if label_array[i] == 6:
				a.append([0,0,0,0,0,1])
	label_array = np.array(a)

	return label_array 

#nn模型
def keras_train_model(prop, mode = 'wf', frequency = 10, std = 0.1, batch_size = 10, layer = 128, ratio = 0.8, drop_1 = 0.2, drop_2 = 0.2, nb_epoch = 10):

	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation
	from keras.optimizers import SGD, Adam, RMSprop
	from keras.utils import np_utils

	if prop == GENDER:
		nb_classes = 2
	else:
		nb_classes = 6
	if mode == 'wf':
		feature_array = np.load(FILE_ROOT + prop + "/word_frequency/feature_" + str(frequency) + ".npy")
	if mode == 'wu':
		feature_array = np.load(FILE_ROOT + prop + "/std/feature_fre_" + str(frequency) + "_std_" + str(std) + ".npy")
	if mode == 'std':
		feature_array = np.load(FILE_ROOT + prop + "/word_user/feature_word_user_" + str(frequency) + ".npy")

	label_array = np.load(FILE_ROOT + prop + "/label.npy")
	label_array = process_label(label_array, prop)
	#label_array = np_utils.to_categorical(label_array, nb_classes)
	print label_array
	feature_length = len(feature_array[0])
	print "feature_length:" + str(feature_length)

	# x_train = feature_array[:int(ratio*len(feature_array))]
	# x_test = feature_array[int(ratio*len(feature_array)):]
	# y_train = label_array[:int(ratio*len(label_array))]
	# y_test = label_array[int(ratio*len(label_array)):]
	x_train = feature_array
	y_train = label_array
	x_data_length = len(x_train)
	print x_data_length

	model = Sequential()
	#第一层输入层
	model.add(Dense(layer, input_shape=(feature_length,)))
	model.add(Activation('relu'))
	model.add(Dropout(drop_1))
	#第二层隐藏层
	model.add(Dense(layer))
	model.add(Activation('relu'))
	model.add(Dropout(drop_2))
	# #第三层隐藏层
	# model.add(Dense(layer))
	# model.add(Activation('relu'))
	# model.add(Dropout(drop_2))
	#输出层
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

	history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=2, validation_split = ratio)
	#score = model.evaluate(x_test, y_test, verbose=0)
	#model.save(FILE_ROOT + prop + "/feature_" + str(frequency) +"_" + str(ratio) + 'keras_model.h5') 
	#print('Test score:', score[0])
	#print('Test accuracy:', score[1])
	model.save(FILE_ROOT + prop + "/feature_" + str(frequency) +"_" + "_batch_" + str(batch_size) + "_layer_" + str(layer) + '_keras_model.h5') 
	#line = prop + "_fre_"+str(frequency)+"_ratio_" + str(ratio) + "_batch_" + str(batch_size) + "_layer_" + str(layer) + "_result accuracy: " + str(score[1]) + " and score: " + str(score[0]) + "\n"
	#print line
	#with open(FILE_ROOT + prop + "/wf_mpl_result.txt", 'a') as f:
	#	f.write(line)
if __name__ == "__main__": 

	#修改编码
	reload(sys)
	sys.setdefaultencoding("utf-8")
	#宏变量
	FILE_ROOT = "E:/Sogou_Project/ccf/data/"
	AGE = "age"
	GENDER = "gender"
	EDUCATION = "education"

	print "processing..."


	keras_train_model(GENDER, mode='wf', frequency=50, std=0.4, batch_size=100, layer=128, ratio = 0.1, drop_1 = 0.5, drop_2 = 0.5, nb_epoch = 10) #acc 84.4 !!!!!


	#测试集
	#get_word_user_test_feature(GENDER)
	#get_word_user_test_feature(GENDER)
	#get_word_user_test_feature(AGE)
	#get_word_user_test_feature(EDUCATION)


	#nn
	# keras_train_model(GENDER, 100, 1, 4096, ratio = 0.8)
	# keras_train_model(GENDER, 100, 10, 4096, ratio = 0.8)
	# keras_train_model(GENDER, 100, 100, 4096, ratio = 0.8)
	# keras_train_model(GENDER, 100, 500, 4096, ratio = 0.8)
	# keras_train_model(GENDER, 100, 2000, 4096, ratio = 0.8)

	#knn
	#train_wf_knn_model(EDUCATION, 100)
	#train_word_user_knn_model(EDUCATION,100)

	#去除空值
	# filter_NA(GENDER)
	# filter_NA(AGE)
	# filter_NA(EDUCATION)

	#词频
	# get_word_frequency(AGE, 5, 6)
	# get_word_frequency(GENDER, 5, 6)
	# get_word_frequency(EDUCATION, 5, 6)

	# #get_wf_feature(AGE, 1, 6)
	#get_wf_feature(GENDER, 5, 6)
	# #get_wf_feature(EDUCATION, 1, 6)

	#保存
	# save_label(AGE)
	# save_label(GENDER)
	# save_label(EDUCATION)


	#csv文件追加，重新生成要删除之前的csv
	#预处理
	# get_word_user_num(GENDER, 2)
	# # get_word_user_ratio(GENDER,2)
	#生成word_user加std词频
	# get_word_user_std_csv(GENDER, std_start = 4, std_end = 5, fre_start = 1, fre_end = 2)
	# get_fre_std_feature(GENDER, std_start = 4, std_end = 5, fre_start = 1, fre_end = 2)
	#生成word_user词频
	# #get_word_user_csv(GENDER, fre_start = 10 , fre_end = 11)
	#get_word_user_feature(GENDER, fre_start = 10, fre_end = 11)

	# get_word_user_num(AGE, 6)
	# get_word_user_ratio(AGE,6)
	# get_word_user_sth(AGE, 6, 3)
	# get_fre_std_feature(AGE, 6, 3)
	# get_word_user_csv(AGE, 6)
	# get_word_user_feature(AGE, 6)

	# get_word_user_num(EDUCATION, 6)
	# get_word_user_ratio(EDUCATION,6)
	# get_word_user_sth(EDUCATION, 6, 3)
	# get_fre_std_feature(EDUCATION, 6, 3)
	# get_word_user_csv(EDUCATION, 6)
	# get_word_user_feature(EDUCATION, 6)