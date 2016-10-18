#encoding=utf-8

import sys
import os
import numpy as np 
import codecs

def findtime(date1,date2):
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


def loadfile(filename):
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
			datedelta=findtime(str(temp[-1]),str(temp[-2]))
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

def writeout(filename,data_list):
	fout=open(filename,'a+')
	for i in range(len(data_list)):
		for j in range(len(data_list[i])):
			fout.write(str(data_list[i][j])+'\t')
		fout.write('\n')
	fout.close()


result=loadfile('ccf_offline_stage1_train.csv')
writeout('normal_ccf_offline_stage1_train.csv',result)

