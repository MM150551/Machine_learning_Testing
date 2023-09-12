import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_excel('titanic.xls')
df.drop(['body','name'],axis=1,inplace=True)
df.fillna(0, inplace=True)


def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
    

    
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))

    return df

# print(handle_non_numeric_data(df))


df = handle_non_numeric_data(df)
y = np.array(df["survived"])
X = np.array(df.drop(['survived'], axis=1))

# print(X)
# print(y)

clf = KMeans(n_clusters=2)
clf.fit(X)
labels = clf.labels_


colors = ["g.","r.","c.","b.","k."]

correct = 0
for i in range(len(X)):
    plt.plot(i , labels[i], colors[labels[i]], markersize = 7)
    plt.plot(i, y[i], marker = 'x')
    if labels[i] == y[i]:
        correct += 1

accuracy = correct/len(X)
print()
print('Accuracy is' , accuracy*100 , '%')
    

plt.show()


