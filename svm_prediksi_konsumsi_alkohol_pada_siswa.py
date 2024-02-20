import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

"""**Memuat data student_mat**"""

student_mat = pd.read_csv('student-mat.csv')
student_mat

"""membandingkan Dalc, Walc, dan kesehatan dengan masing-masing usia, pertama-tama mengelompokkan usia dan mendapatkan konsumsi alkohol maksimal, min, dan min di student_math"""

display(student_mat[["school","sex","age","Dalc","Walc","health",]].groupby(["age"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="Oranges_r"))

"""Menambahkan kolom baru, label untuk student_mat :1"""

student_mat['label']="1"

student_mat.head()

"""**Memuat data student_por**"""

student_por = pd.read_csv('student-por.csv')
student_por

"""membandingkan Dalc, Walc, dan kesehatan dengan masing-masing usia, pertama-tama mengelompokkan usia dan mendapatkan konsumsi alkohol maksimal, min, dan min di student_por"""

display(student_por[["school","sex","age","Dalc","Walc","health",]].groupby(["age"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="Oranges"))

"""Menambahkan kolom baru, label untuk student_por :0"""

student_por['label']="0"
student_por.head()

"""Merge Dataset"""

Data= student_mat.append([student_mat,student_por])
x = Data.iloc[:, [3]].values
Data

"""Prediksi"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def FunLabelEncoder(df):
    for c in df.columns:
        if df.dtypes[c] == object:
            le.fit(df[c].astype(str))
            df[c] = le.transform(df[c].astype(str))
    return df

Data = FunLabelEncoder(Data)
Data.info()
Data.iloc[0:4,:]

"""Split-ing Data"""

from sklearn.model_selection import train_test_split
Y = Data['label']
X = Data.drop(columns=['label'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=9)

print('X train shape: ', X_train.shape)
print('Y train shape: ', Y_train.shape)
print('X test shape: ', X_test.shape)
print('Y test shape: ', Y_test.shape)

"""Data Train"""

X_train

Y_train

"""Data Test"""

X_test

Y_test

"""# **Klasifikasi SVM**"""

from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# mendefinisikan model SVM
svmcla = OneVsRestClassifier(BaggingClassifier(SVC(C=10,kernel='rbf',random_state=9, probability=True),
                                               n_jobs=-1))

# melatih model
svmcla.fit(X_train, Y_train)

# memprediksi target nilai
Y_predict2 = svmcla.predict(X_test)

test_acc_svmcla = round(svmcla.fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)
train_acc_svmcla = round(svmcla.fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)

# Matrik permasalahan/problem
svmcla = confusion_matrix(Y_test, Y_predict2)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(svmcla, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
plt.title('SVM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

"""Akurasi Data / Ketepatan"""

model = pd.DataFrame({
    'Model': ['SVM'],
    'Train Score': [train_acc_svmcla],
    'Test Score': [test_acc_svmcla]
})
model.sort_values(by='Test Score', ascending=False)

"""Skor perolehan presisi rata-rata"""

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(Y_test, Y_predict2)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))