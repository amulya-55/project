# project
Email spam detection using Machine Learning

program:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the data from csv file
spam = pd.read_csv('/content/spam.csv',encoding='latin-1')
print(spam)

#replace the null values with a null string
spam = spam.where((pd.notnull(spam)),'')

#printing the first 5 rows of df
spam.head()pam = spam.where((pd.notnull(spam)),'')

#checking the number of rows and columns in df
spam.shape

#Label spam mail as 0; ham mail as 1;
spam.loc[spam['v1'] == 'spam', 'v1',] = 0
spam.loc[spam['v1'] == 'ham', 'v1',] = 1

x = spam['v2']
y = spam['v1']

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)

print(x.shape)
print(x_train.shape)
print(x_test.shape)

#transform the text data to feature vectors that can be used as input in Logistic Regession
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

#convert y_train and Y-test values as integer
y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(x_train)

print(x_train_features)

model = LogisticRegression()

#training the Logistic Regression model with the training data
model.fit(x_train_features,y_train)

#prediction on training data
Prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train,Prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

#prediction on test data
Prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test,Prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

#Building a predictive system
input_mail = ["I've been searching for the right words to thank your this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

#convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

#making prediction
prediction = model.predict(input_data_features)
print(prediction)

if (prediction[0]==1):
  print('Ham mail')
else:
  print('Spam mail')
