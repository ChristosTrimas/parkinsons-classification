"""
I will try to classify the parkinson dataset patients with the following classifiers
1)Support Vector Classifier
2)Naive Bayes 
3)Logistic Regression 
4)Stochastic Gradient Descent 
5)K-Nearest Neighbors
6)Random Forest Classifier 
"""
import fileinput
import pandas as pd
import os , sys, warnings
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from warnings import simplefilter
from sklearn.exceptions import DataConversionWarning

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
print("Setup Ready!")

#loading the dataset
ps = pd.read_csv('/home/broly/Desktop/parkinsons.data')
#loading features and labels and encode them as. park: patients with parkinson, no_park: patients with no parkinson
features = ps.loc[:,ps.columns!='status'].values[:,1:]
labels = ps.loc[:,'status'].values
group_names = ['no_park','park']
ps['status'] = pd.cut(ps['status'],bins=2,labels=group_names)
ps['status'].unique()

"""
Splitiing the data to train and test data.
I also scale the data so that features with larger values do not impact more than those with smaller.
"""

X_train, X_test,y_train,y_test = train_test_split(features,labels, test_size = 0.20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


################################################################
#					Random Forest Classifier                   #
################################################################
def rfc():
	print("Random Forest Classifier")
	model_1 = RandomForestClassifier(n_estimators=200)
	model_1.fit(X_train,y_train)
	m1_prediction = model_1.predict(X_test)

	# performance
	print(classification_report(y_test,m1_prediction))
	print(confusion_matrix(y_test,m1_prediction))
	print(accuracy_score(y_test,m1_prediction)*100)


################################################################
#					Support Vector Classifier 				   #
################################################################
def svc(select):
	
	if select=='Linear':
		print("Linear kernel Support Vector Classifier")
		model_2 = svm.SVC(kernel='linear')
		model_2.fit(X_train,y_train)
		m2_prediction = model_2.predict(X_test)

		# performance
		print(classification_report(y_test,m2_prediction))
		print(confusion_matrix(y_test,m2_prediction))
		print(accuracy_score(y_test,m2_prediction)*100)

	if select=='Polynomial':
		print("Poly kernel Support Vector Classifier")
		model_3 = svm.SVC(kernel='poly',degree=4)
		model_3.fit(X_train,y_train)
		m3_prediction = model_3.predict(X_test)

		# performance
		print(classification_report(y_test,m3_prediction))
		print(confusion_matrix(y_test,m3_prediction))
		print(accuracy_score(y_test,m3_prediction)*100)


	if select=='Sigmoid':
		print("Sigmoid kernel Support Vector Classifier")
		model_4 = svm.SVC(kernel='sigmoid')
		model_4.fit(X_train,y_train)
		m4_prediction = model_4.predict(X_test)

		# performance
		print(classification_report(y_test,m4_prediction))
		print(confusion_matrix(y_test,m4_prediction))
		print(accuracy_score(y_test,m4_prediction)*100)

############################################################
#					Naive Bayes Classifier				   #
############################################################
def bayes():
	print("Naive Bayes Classifier")

	model_5 = GaussianNB()
	model_5.fit(X_train,y_train)
	m5_prediction = model_5.predict(X_test)


	print(classification_report(y_test,m5_prediction))
	print(confusion_matrix(y_test,m5_prediction))
	print(accuracy_score(y_test,m5_prediction)*100)

############################################################
#					Logistic Regression 				   #
############################################################
def LogisticReg():
	print("Logistic Regression Classifier")
	model_6 = LogisticRegression(random_state=0).fit(X_train,y_train)
	m6_prediction = model_6.predict(X_test)

	print(classification_report(y_test,m6_prediction))
	print(confusion_matrix(y_test,m6_prediction))
	print(accuracy_score(y_test,m6_prediction)*100)


#############################################################
#					Stochastic Gradient Descent Classifier  #
#############################################################
def SGDCl():

	print("Stochastic Gradient Descent Classifier")
	model_7 = SGDClassifier(loss='perceptron',shuffle=True,penalty='l1').fit(X_train,y_train)
	m7_prediction = model_7.predict(X_test)

	print(classification_report(y_test,m7_prediction))
	print(confusion_matrix(y_test,m7_prediction))
	print(accuracy_score(y_test,m7_prediction)*100)


##############################################################
#					K NN Classifier 						 #
##############################################################
def knncl():
	for i in range(2,10):
		print(f"{i}-NN Classifier")
		model_8 = KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)
		m8_prediction = model_8.predict(X_test)

		print(classification_report(y_test,m8_prediction))
		print(confusion_matrix(y_test,m8_prediction))
		print(accuracy_score(y_test,m8_prediction)*100)



def main():

	start = True
	while start:
		
		print("=====================================================")
		print("==========  The dataset is loaded. ==================")
		print("==========  Choose which classification you want: ===")
		print("=========== 1: Random Forest ========================")
		print("=========== 2: K NN =================================")
		print("=========== 3: Naive Bayes ==========================")
		print("=========== 4: Stochastic Gradient Descent ==========")
		print("=========== 5: Logistic Regression ==================")
		print("=========== 6: Support Vector =======================")
		print("=========== 7: Exit =================================")
		print("=====================================================")

		select = input("Press number: ")
		
		if select=='1':
			rfc()
		elif select=='2':
			knncl()
		elif select=='3':
			bayes()
		elif select=='4':
			SGDCl()
		elif select=='5':
			LogisticReg()
		elif select=='6':
			print("What type of SVC you want?")
			print("1: Linear")
			print("2: Polynomial")
			print("3: Sigmoid")
			choice = input()
			if choice=='Linear' or choice=='Polynomial' or choice=='Sigmoid':
				svc(choice)
			else:
				print("Wrong choice")
		elif select=='7':
			start = False 
		else:
		 	print("try again")

main()