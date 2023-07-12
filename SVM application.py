import numpy as np
from sklearn import preprocessing, model_selection, neighbors , svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop('id',inplace=True, axis=1)
# print(df)

X=np.array(df.drop(['class'],axis=1))
y=np.array(df['class'])

X_train , X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

clf = svm.SVC(gamma="auto")
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,2,1,3]])
#example_measures = example_measures.reshape(len(example_measures),-1)

print(clf.predict(example_measures))