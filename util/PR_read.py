# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-06-15 23:08
@Project:Personal_Recommendation_Algorithm
@Filename:read.py
"""

import os

def get_graph_from_data(input_file):
	"""
	:param input_file: user item rating file
	:return:
		a dict:{UserA:{itemb:1,itemc:1},itemb:{UserA:1}}
	"""
	if not os.path.exists(input_file):
		return {}
	graph = {}
	score_thr = 4.0
	fp = open(input_file,encoding='utf-8')
	for line in fp:
		item = line.strip().split("::")
		if len(item) < 3:
			continue
		userid,itemid,rating = item[0],"item_" + item[1],item[2]
		if float(rating) < score_thr:
			continue
		if userid not in graph:
			graph[userid] = {}
		graph[userid][itemid] = 1
		if itemid not in graph:
			graph[itemid] = {}
		graph[itemid][userid] = 1
	fp.close()
	return graph




if __name__ == "__main__":
	graph = get_graph_from_data("../data/ratings.dat")
	print(graph['1'])
