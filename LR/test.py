# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-06-19 16:09
@Project:Personal_Recommendation_Algorithm
@Filename:test.py
"""

if __name__ == "__main__":
	fp = open("../data/train_file")
	count = 0
	for line in fp:
		item = line.strip().split(",")
		print(len(item))
		count += 1
		if count >= 10:
			break