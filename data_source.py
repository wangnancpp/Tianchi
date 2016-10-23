#encoding=utf-8

import sys
import os
import numpy as np 
import codecs
import random
import datetime
import data_statistics
from sklearn import cross_validation
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
import model_train
import matplotlib.pyplot as plt

class DataSource:
	
	def findtime(self,date1,date2):
		sum1=0
		sum2=0
		try:
			day1=int(date1[-2:])
			day2=int(date2[-2:])
			month1=int(date1[-4:-2])
			month2=int(date2[-4:-2])
			sum1=month1*30+day1
			sum2=month2*30+day2
			return sum1-sum2
		except:
			return 0

	# base date for date normalize
	base_date = datetime.datetime(2016, 01, 01)
	def get_days(self, date_str):
		d = datetime.datetime.strptime(date_str,'%Y%m%d')
		return (d - self.base_date).days
		
	def splitlist(self, result, result1):
		result_sum=[]
		for i in range(len(result)):
			if result[i] in result1:
				continue
			else:
				result_sum.append(result[i])
		print "split list done"
		return result_sum
		
	# split data into train data and test data
	def split_data(self, total_list, train_ratio):
		data_size = len(total_list)
		if data_size <= 0:
			return
		train_list = []
		test_list = []
		train_size = int(data_size * train_ratio)
		train_index = random.sample(xrange(data_size), train_size)
		train_index_set = set(train_index)
		
		train_list = []
		test_list = []
		for i in xrange(data_size):
			if i in train_index_set:
				train_list.append(total_list[i])
			else:
				test_list.append(total_list[i])

		return (train_list, test_list)
		
	def loadfile(self,filename):
		fin=codecs.open(filename,'r','utf-8')
		lines=fin.readlines()
		result=[]
		for line in lines:
			temp=[]
			coupon=True
			line=line.strip().split(',')
			for i in range(len(line)):
				if i==0 or i==1 or i==6:
					temp.append(str(line[i]))
				elif(i==2):
					if line[i]=='null':
						coupon=False
						temp.append('null')
					else:
						temp.append(line[i])
				elif i==3 or i==5:
					if coupon:
						temp.append(str(line[i]))
					else:
						temp.append('null')
				elif i==4:
					if line[i]=='null':
						temp.append('null')
					else:
						num_int=int(line[i])
						if num_int==0:
							temp.append(50)
						else:
							temp.append(num_int*500)
			if temp[-1]!='null' and temp[-2]!='null':
				datedelta = self.findtime(str(temp[-1]),str(temp[-2]))
				temp.append(datedelta)
			elif temp[-1]!='null' and temp[-2]=='null':
				temp.append(100)
			elif temp[-1]=='null' and temp[-2]=='null':
				temp.append('0:0')
			elif temp[-1]=='null' and temp[-2]!='null':
				temp.append(-1)
			result.append(temp)
		fin.close()
		return result

	# load origin data, data type is numpy
	def load_data(self, path):
		return np.loadtxt(path, dtype=np.str, delimiter=",")

	# data with mocked label
	def load_normalize_data(self, path):
		# all offline data load
		origin_data = self.load_data(path)
		return self.normalize_data(origin_data)

	def generate_label(self, row):
		if len(row) > 6:
			date = row[6]
			if date != 'null':
				return 1
			else:
				return 0
		else:
			return -1

	def normalize_data(self, origin_data):
        
		# num of rows
		rows = origin_data.shape[0]
		# labels 
		labels =np.arange(rows, dtype = np.int) #  np.empty([rows, 1], dtype = np.float)
		
		# features array
		features = np.empty([rows, 9], dtype = np.int)
		
		#load stat data
		data_stat = data_statistics.Stat()
		data_stat.stat(origin_data)
		
		user_stat = data_stat.data_stat["users"]
		merchant_stat = data_stat.data_stat["merchants"]
		coupon_stat = data_stat.data_stat["coupons"]
		
		for i in xrange(len(origin_data)):
			row = origin_data[i]
			
			uid = row[0]
			mid = row[1]
			coupon_id = row[2]
			discount_rate = row[3]
			distance = row[4]
			date_received = row[5]
			
			#set label
			labels[i] = self.generate_label(row)
			
			# to get feature
			feature = features[i]
			
			# discount_rate
			f_discount_rate = -1.0
			if discount_rate != 'null':
				try:
					if discount_rate.find(":") >= 0:
						strs = discount_rate.split(':')
						f_discount_rate = float(strs[1]) / float(strs[0])
						feature[-1] = float(strs[0])
					else:
						f_discount_rate = float(discount_rate)
							
				except:
					feature[-1] = 0
			
			# distance	
			f_distance = -1.0
			if distance != "null":
				f_distance = float(distance)

			# collect feature
			
			feature[0] = f_discount_rate
			feature[1] = f_distance
			
			# user feature
			user_coupons = -1
			user_consume = -1
			# if uid in user_stat:
			# 	user_stat_item = user_stat[uid]
			# 	if "num_coupons" in user_stat_item:
			# 		user_coupons = user_stat_item["num_coupons"]
			# 	if "num_consumes" in user_stat_item:
			# 		user_consume = user_stat_item["num_consumes"]
				 
			feature[2] = user_coupons
			feature[3] = user_consume
			
			# merchant feature
			merchant_coupons = -1
			merchant_consume = -1
			if mid in merchant_stat:
				merchant_stat_item = merchant_stat[mid]
				if "num_coupons" in merchant_stat_item:
					merchant_coupons = merchant_stat_item["num_coupons"]
				if "num_consumes" in merchant_stat_item:
					merchant_consume = merchant_stat_item["num_consumes"]
					
			feature[4] = merchant_coupons
			feature[5] = merchant_coupons
			
			# coupon feature
			coupon_coupons = -1
			coupon_consume = -1
			# if coupon_id in coupon_stat:
			# 	coupon_stat_item = coupon_stat[coupon_id]
			# 	if "num_coupons" in coupon_stat_item:
			# 		coupon_coupons = coupon_stat_item["num_coupons"]
			# 	if "num_consumes" in coupon_stat_item:
			# 		coupon_consume = coupon_stat_item["num_consumes"]
			
			feature[6] = coupon_coupons
			feature[7] = coupon_consume
			
			# 
		return (features, labels)

	def writeout(self,filename,data_list):
		print "start writing"
		fout=open(filename,'a+')
		for i in range(len(data_list)):
			for j in range(len(data_list[i])):
				fout.write(str(data_list[i][j])+'\t')
			fout.write('\n')
		fout.close()

def generate_test_result():
	# train with all offline data
	train_file = "ccf_offline_stage1_train.csv"
	test_file = "ccf_offline_stage1_test_revised.csv"
	data_source = DataSource()
	(train_features, train_labels) = data_source.load_normalize_data(train_file)
	(test_features, test_labels) = data_source.load_normalize_data(test_file)
	
	#
	model = model_train.Model()
	model.train(train_features, train_labels)
	#model.predict(test_features, test_labels)
	predicted_prob = model.model.predict_proba(test_features)
	
	fout = open("test.label.out", "w")
	test_lines = open(test_file).readlines()
	for i in xrange(len(test_lines)):
		line = test_lines[i].strip()
		fout.write(line + "," + str(round(predicted_prob[i][1], 1)) + "\n")
	fout.close()
		
	
if __name__=='__main__':
	#generate_test_result()
	#sys.exit()
	
	input_file_name = "ccf_offline_stage1_train.csv"
	data_source = DataSource()
	all_data = data_source.load_data(input_file_name)
	
	#split data into train and test

	# train_list = []
	# test_list = []
	# for item in all_data:
	# 	if item[5] != "null":
	# 		if data_source.get_days(item[5]) > 150:
	# 			test_list.append(item)
	# 		else:
	# 			train_list.append(item)
	# 	else:
	# 		test_list.append(item)
			
	# X_train = np.array(train_list)
	# X_test = np.array(test_list)
	
	(X_train, X_test, y_train, y_test) = cross_validation.train_test_split(all_data, 
		np.arange(all_data.shape[0]), test_size=0.3)
	# 
	(train_features, train_labels) = data_source.normalize_data(X_train)
	(test_features, test_labels) = data_source.normalize_data(X_test)
	
	print train_labels
	print train_features.shape
	print train_labels.shape
	print test_features.shape
	print test_labels.shape
	
	model = model_train.Model()
	model.train(train_features, train_labels)
	model.predict(test_features, test_labels)
	
	predicted_prob = model.model.predict_proba(test_features)
	
	# Plot outputs
	# plt.scatter(test_features, test_labels,  color='black')
	# plt.plot(test_features, model.model.predict(test_features), color='blue',
	# 		linewidth=3)

	# plt.xticks(())
	# plt.yticks(())

	# plt.show()
	