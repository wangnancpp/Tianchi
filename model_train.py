import data_source
import sys
import numpy as np
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn import cross_validation

class Model:
	model = None
	train_data = None
	test_data = None
	
	def __init__(self):
		self.model = LogisticRegression()
		#self.model = svm.SVC(kernel='linear')
	
	def load_data(self, train_data, test_data):
		self.train_data = train_data
		self.test_data = test_data
	
	def train(self, train_data, label_data):
		self.model.fit(train_data, label_data)
		
	def predict(self, test_data, test_label):
		expected = test_label
		predicted = self.model.predict(test_data)
		print(metrics.classification_report(expected, predicted))
		print(metrics.confusion_matrix(expected, predicted))

		
if __name__ == "__main__":
	file_name = "ccf_offline_stage1_train.csv"
	data  = data_source.DataSource()
	(features, labels) = data.load_normalize_data(file_name)
	print features
	print labels
	#sys.exit()
	(X_train, X_test, y_train, y_test) = cross_validation.train_test_split(features, labels, test_size=0.4)
	 
	model  = Model()
	model.train(X_train, y_train)
	model.predict(X_test, y_test)
	 