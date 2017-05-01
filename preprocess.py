
# -*- coding:utf-8 -*-  
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import re
import sys
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba.posseg as pseg

#读原始数据
def process_origin_data(orgin_file_name, tag_data_file_name, text_data_file_name, sep):
	#注意编码问题
	origin_data_file = open(orgin_file_name)
	tag_data_flie = open(tag_data_file_name, 'w')
	text_data_file = open(text_data_file_name, 'w')
	tag_lines = []
	text_lines = []
	while True:
		line = origin_data_file.readline()
		if not line:
			origin_data_file.close()
			break
		else:
			print(line[:sep])
			tag_lines.append(line[:sep] + "\n")
			text_lines.append(line[sep:])
	tag_data_flie.writelines(tag_lines)
	text_data_file.writelines(text_lines)
	#关闭文件
	tag_data_flie.close()
	text_data_file.close()


#分词
def jieba_sep(text_name, sep_text_name):

	text = open(text_name)
	sep_text = open(sep_text_name, 'w')
	sep_text_list =[]
	i=0
	#读停用词
	stopwords = {}.fromkeys([line.rstrip() for line in open(r'./data/stopwords.txt')])
	print len(stopwords)
	if "双沟" in stopwords:
		print "OK!!!!!!!!!!!!!!!!!!!"

	while True:
		line = text.readline()
		#print line
		pattern = re.compile(r"\t")
		line = re.sub(pattern, '', line, count = 1)
		line = ' '.join(line.split())

		#print line
		if not line:
			text.close()
			break
		else:
			i += 1
			
			#seg_all_list = jieba.cut_for_search(line)
			seg_all_list = pseg.cut(line)
			seg_list = []
			for word, flag in seg_all_list:
				#print word , flag
				if word.encode('utf-8') not in stopwords:
					#只保留非地名的名词
					if flag[0] == 'n' and flag != "ns" :
						seg_list.append(word)
						#print word , flag
			final_words = " ".join(seg_list)

			pattern = re.compile(r'\w*', re.L)
			final_words = re.sub(pattern, '', final_words)
			final_words_list = final_words.split()
			final_words = ""
			#去单字
			for word in final_words_list:
				if len(word) > 1 :
					final_words += word + " "

			#seg_key_words_list = jieba.analyse.extract_tags(final_words, KEY_WORDS_NUM)
			#final_words = " ".join(seg_key_words_list)

			print("jieba ing " + str(i) +": " + final_words)
			
			sep_text_list.append(final_words + "\n")
	print(len(sep_text_list))
	#保存
	sep_text.writelines(sep_text_list)
	sep_text.close()

def merge_tag_and_words(tag_file, word_file):
	#加载标签
	tag_df = pd.read_csv(tag_file, sep='	', header=None, names = ["ID", "age", "gender", "education"])
	#print(tag_df.head())
	#加载分词
	words_df = pd.read_csv(word_file, header=None,  names=['content'])
	#合并
	all_info_df = tag_df
	print(len(tag_df), len(words_df))
	all_info_df['content'] =""
	all_info_df['content'] = words_df['content']
	df = all_info_df[all_info_df['content'] == "英文单词"]
	print(df[['ID']])
	print(all_info_df.head())
	all_info_df.to_csv("train_all_info.csv", sep =',' ,index=None)

	#print(words_df.head())

#获取每个用户的搜索词列表
def get_split_word(age_df):
	#class_df = pd.read_csv(file_root + "class/" + class_file, names=["word", "class"], sep=' ')
	#all_info_df = pd.read_csv(file_root+ "train_new_all_information.csv", na_values = 0)
	#age_df = all_info_df[[kind, 'content']]
	#age_df = age_df.dropna()
	#a = age_df[[kind]].values
	#np.save(file_root+kind +"_class_tags", a)
	#获取训练集的搜索词
	words = age_df[['content']].values
	#将搜索词串放入列表
	word_list = []
	for i in range(len(words)):
		word_list.append(words[i][0])
	#搜索词串分割
	split_word_list = []
	for words in word_list:
		cur_split = words.split()
		split_word_list.append(cur_split)
	return split_word_list


#先分词，根据类别分词，先去除na，再算词频。先根据词频做训练。
if __name__ == "__main__": 
	#df = pd.read_csv("C:/Users/YYC/Desktop/ccf/user_tag_query.2W.train.csv", sep='|')
	#print(df)
	reload(sys)
	sys.setdefaultencoding("utf-8")
	print "start"
	KEY_WORDS_NUM = 20
	TEST_SEP = 32
	TRAIN_SEP = 38
	file_root = r"E:\Sogou_Project\ccf"


	#处理原始测试数据集
	#process_origin_data(file_root+"/data/origin/user_tag_query.10W.TEST.csv", file_root+"/data/10W.new_test.ID.csv", file_root+"/data/10W.new_test.text.csv", TEST_SEP)
	#处理原始训练数据集
	#process_origin_data(file_root+"/data/origin/user_tag_query.10W.TRAIN.csv", file_root+"/data/10W.new_train.TAG.csv", file_root+"/data/10W.new_train.text.csv", TRAIN_SEP)

	#测试集文本分词
	#jieba_sep(file_root+"/data/10W.new_test.text.csv", file_root+"/data/10W.new_test.word.csv")
	#训练集文本分词
	#jieba_sep(file_root+"/data/10W.new_train.text.csv", file_root+"/data/10W.train.word.noun.no.place.csv")
	#合并标签和数据集
	merge_tag_and_words(file_root+"/data/10W.new_train.TAG.csv", file_root+"/data/10W.train.word.noun.no.place.csv")
	#merge_tag_and_words(file_root+"2W.new_train.TAG.csv", file_root+"replace_train_content.csv")
	#merge_tag_and_words(file_root+"2W.new_train.TAG.csv", file_root+"replace_word.csv")




	