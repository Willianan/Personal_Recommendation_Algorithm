# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-06-19 21:25
@Project:Personal_Recommendation_Algorithm
@Filename:get_feature_num.py
"""

import os
def get_feature_num(feature_num_file):
	if not os.path.exists(feature_num_file):
		return 0
	else:
		fp = open(feature_num_file,encoding="utf-8")
		for line in fp:
			item = line.strip().split("=")
			if item[0] == "feature_num":
				return int(item[1])
		return 0