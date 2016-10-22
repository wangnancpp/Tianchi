#encoding=utf-8

import sys
import os
import numpy as np 
import codecs
import random

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


	def splitlist(self, result, result1):
		result_sum=[]
		for i in range(len(result)):
			if result[i] in result1:
				continue
			else:
				result_sum.append(result[i])
		print "split list done"
		return result_sum

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



	def writeout(self,filename,data_list):
		print "start writing"
		fout=open(filename,'a+')
		for i in range(len(data_list)):
			for j in range(len(data_list[i])):
				fout.write(str(data_list[i][j])+'\t')
			fout.write('\n')
		fout.close()

if __name__=='__main__':
	train_ratio = 0.7
	print "starting"
	data_source = DataSource()
	result=data_source.loadfile('ccf_offline_stage1_train.csv')
	print "read total data %d lines" %(len(result))
	
	print "start split_data"
	(train_list, test_list) = data_source.split_data(result, train_ratio)
	print "train size %d, test size %d" % (len(train_list), len(test_list))
	sys.exit()
	
	print "the first is :"
	result_test=random.sample(result,int(len(result) * train_ratio))
	print "the second is :"
	result_train=data_source.splitlist(result,result_test)
	writeout('normal_ccf_offline_stage1_train.csv',result_train)
	writeout('normal_ccf_offline_stage1_test.csv',result_test)


