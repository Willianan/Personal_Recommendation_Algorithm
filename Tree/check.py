# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-06-20 14:59
@Project:Personal_Recommendation_Algorithm
@Filename:check.py
@description:
	test the performance of the model in the test data for gbdt and gbdt_lr
"""




from __future__ import division
import numpy as np
import xgboost as xgb
import sys
import Tree.train as TA
import math
from scipy.sparse import csc_matrix
sys.path.append("../util")
import util.get_feature_num as GF




def get_test_data(test_file,feature_num_file):
	"""
		:param test_file: file to check performance
		:param feature_num_file: the file record total num of feature
		:return:
			two np.array:test_feature,test_label
		"""
	total_feature_num = GF.get_feature_num(feature_num_file)
	test_label = np.genfromtxt(test_file, dtype=np.float32, delimiter=",", usecols=-1)
	feature_list = range(total_feature_num)
	test_feature = np.genfromtxt(test_file, dtype=np.float32, delimiter=",", usecols=feature_list)
	return test_feature, test_label

def predict_by_tree(test_feature,tree_model):
	"""
	predict by gbdt model
	"""
	predict_list = tree_model.predict(xgb.DMatrix(test_feature))
	return predict_list

def predict_by_lr_gbdt(test_feature,mix_tree_model,mix_lr_coef,tree_info):
	"""
	predict by mix model
	"""
	tree_leaf = mix_tree_model.predict(xgb.DMatrix(test_feature),pred_leaf=True)
	(tree_depth,tree_num,step_size) = tree_info
	total_feature_list = TA.get_gbdt_and_lr_feature(tree_leaf,tree_depth=tree_depth,tree_num=tree_num)
	result_list = np.dot(csc_matrix(mix_lr_coef),total_feature_list.tocsc().T).toarray()[0]
	sigmoid_ufunc = np.frompyfunc(sigmoid,1,1)
	return sigmoid_ufunc(result_list)

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def get_auc(predict_lsit,test_label):
	"""
	:param predict_lsit: model predict score list
	:param test_label: label of test data
	auc = (sum(pos_index)-pos_num(pos_num +1)/2)/pos_num*neg_num
	"""
	total_list = []
	for index in range(len(predict_lsit)):
		predict_score = predict_lsit[index]
		label = test_label[index]
		total_list.append((label,predict_score))
	sorted_total_list = sorted(total_list,key=lambda ele:ele[1])
	neg_num = 0
	pos_num = 0
	count = 1
	total_pos_index = 0
	for zuhe in sorted_total_list:
		label,predict_score = zuhe:
		if label == 0:
			neg_num += 1
		else:
			pos_num += 1
			total_pos_index += count
		count += 1
	auc_score = (total_pos_index - (pos_num) * (pos_num + 1) / 2) / (pos_num * pos_num)
	print("auc: %.5f" %(auc_score))

def get_accuary(predict_list,test_label):
	"""
	:param test_feature: model predict score list
	:param test_label:  label of test data
	"""
	score_thr = 0.8
	right_num = 0
	for index in range(len(predict_list)):
		predict_score = predict_list[index]
		if predict_score >= score_thr:
			predict_label = 1
		else:
			predict_label = 0
		if predict_label  == test_label[index]:
			right_num += 1
	total_num = len(predict_list)
	accuary_score = right_num / total_num
	print("accuary: %.5f" %(accuary_score))

def run_check_core(test_feature,test_label,model,score_func):
	"""
	:param test_feature:
	:param test_label:
	:param model: tree model
	:param score_func: use different model to predict
	"""
	predict_list = score_func(test_feature,model)
	get_auc(predict_list,test_label)
	get_accuary(predict_list,test_label)

def run_check(test_file,test_model_file,feature_num_file):
	"""
	:param test_file: file to test performance
	:param test_model_file: gbdt model dile
	:param feature_num_file: file to store feature num
	"""
	test_feature,test_label = get_test_data(test_file,feature_num_file)
	tree_model = xgb.Booster(model_file=test_model_file)
	run_check_core(test_feature,test_label,tree_model,predict_by_tree)


def run_check_lr_gbdt_core(test_feature, test_label, mix_tree_model, mix_lr_coef,tree_info, score_func):
	"""
	core function to test the performance of the mix model
	:param test_feature:
	:param test_label:
	:param mix_tree_model:
	:param mix_lr_coef:
	:param tree_info: tree_depth, tree_num, learning_rate
	:param score_func: different score_func
	:return:
	"""
	predict_list = score_func(test_feature,test_label,mix_tree_model,mix_lr_coef,tree_info)
	get_auc(predict_list,test_label)
	get_accuary(predict_list,test_label)



def run_check_lr_gbdt(test_file,tree_mix_model_file,lr_coef_mix_model_file,feature_num_file):
	"""
	:param test_file:
	:param tree_mix_model_file: tree part of mix model
	:param lr_coef_mix_model_file: lr part of mix model
	:param feature_num_file:
	"""
	test_feature,test_label = get_test_data(test_file,feature_num_file)
	mix_tree_model = xgb.Booster(model_file=tree_mix_model_file)
	mix_lr_coef = np.genfromtxt(lr_coef_mix_model_file,dtype=np.float32,delimiter=",")
	tree_info = TA.get_mix_model_tree_info()
	run_check_lr_gbdt_core(test_feature,test_label,mix_tree_model,mix_lr_coef,tree_info,predict_by_lr_lr_gbdt)




if __name__ == "__main__":
	run_check("../data/test_file","../data/xgb.model","../data/feature_num")
	run_check_lr_gbdt("../data/test_file","../data/xgb_mix_model","../data/lr_coef_mix_model","../data/feature_num")