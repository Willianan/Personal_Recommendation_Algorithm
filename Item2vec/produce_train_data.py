# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-06-16 16:37
@Project:Personal_Recommendation_Algorithm
@Filename:produce_train_data.py
"""


import os
import sys

def produce_train_data(input_file,out_file):
	"""
	:param input_file: user behavior file
	:param out_file: output file
	"""
	if not os.path.exists(input_file):
		return
	record = {}
	score_thr = 4.0
	fp = open(input_file,encoding='utf-8')
	for line in fp:
		item = line.strip().split("::")
		if len(item) < 4:
			continue
		userid,itemid,rating = item[0],item[1],float(item[2])
		if rating < score_thr:
			continue
		if userid not in record:
			record[userid] = []
		record[userid].append(itemid)
	fp.close()
	fw = open(out_file,'w+')
	num = 0
	for userid in record:
		if num > 1000:
			break
		num += 1
		fw.write("::".join(record[userid]) + "\n")
	fw.close()




if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("usage: python xx.py inputfile outputfile")
		sys.exit()
	else:
		inputfile = sys.argv[1]
		outputfile = sys.argv[2]
		produce_train_data(inputfile,outputfile)
	#produce_train_data("../data/ratings.dat","../data/train_data.txt")