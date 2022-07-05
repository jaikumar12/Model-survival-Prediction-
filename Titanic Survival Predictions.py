#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


### Read csv files
data = pd.read_csv('train.csv')


# In[3]:


## Print dataset
data


# In[4]:


### Check dataset info

data.info()


# In[5]:


### Count total missing values in dataset

data.isnull().sum()


# # In this dataset column Cabin missing 70% data . So we can drop the cabin in this dataset

# In[6]:


data= data.drop(['Cabin'],axis=1)


# In[7]:


data


# In[8]:


### Number of rows and columns

data.shape


# In[9]:


# Replacing the missing values in 'Age' with mean values

data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[10]:


## Finding the mode values in 'Embarked' columns
data['Embarked'].mode()


# In[11]:


# Replacing the missing values in 'Embarked' column with the mode

data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)


# In[12]:


## Again check the missing values

data.isnull().sum()


# # EDA 

# In[13]:


## Finding the number of people survived or not survived

data['Survived'].value_counts()


# In[14]:


## Making a countplot for 'Survived' column
plt.figure(figsize=(7,5))
sns.countplot('Survived',data=data)


# # Oberving Survived column we know that number of survived is less than number of not survived

# In[15]:


## Finding number of Male And Female in 'Sex' column
data['Sex'].value_counts()


# In[16]:


## Making a countplot for 'sex' with respect 'Survived' 
plt.figure(figsize=(7,7))
sns.countplot('Sex',hue='Survived',data=data)


# # By looking the graph we  can understand the number of survived Female is more than Male

# In[17]:


## Finding the number of people in 'Pclass'

data['Pclass'].value_counts()


# In[18]:


## Making a countplot for 'Pclass' with respect Survived

plt.figure(figsize=(7,7))
sns.countplot('Pclass',hue='Survived',data=data)


# # By looking the graph we find the number of not survived is more in pclass 3 as compare the both class
# # Number of survived is more in pclass 1 as compare another two pcalss

# In[19]:


### Finding the number of people in 'Embarked'

data['Embarked'].value_counts()


# In[20]:


### Making a countplot for 'Embarked' with respect Survived

plt.figure(figsize=(7,7))
sns.countplot('Embarked',hue='Survived',data=data)


# # By looking the graph the find number of survived or not survived is more in 'S' Embarked as compare to another Embarked

# In[21]:


### Converting categorical Columns

data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[22]:


data


# In[33]:


### WE can drop some categorical columns because it not much sense

df = data.drop(['PassengerId','Name','Ticket'],axis=1)


# In[42]:


### Data split in X(Feature variable) and y(target variable) 
x = df.iloc[:,1:].values
y = df.iloc[:,0].values


# In[43]:


x


# In[44]:


y


# In[45]:


### Split X and Y into training and test sets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[46]:


### Check the shape of x_train and x_test
x_train.shape,x_test.shape


# # Model Training

# In[47]:


## Train a logistic regression model on the training set

from sklearn.linear_model import LogisticRegression


# In[48]:


### Instantiate the model

logreg = LogisticRegression()


# In[49]:


### Fit the model

logreg.fit(x_train,y_train)


# # Finding Accuracy  

# In[50]:


x_train_pred = logreg.predict(x_train)


# In[51]:


x_train_pred


# In[52]:


y_pred1 = logreg.predict_proba(x_test)[:,1]


# In[55]:


from sklearn.metrics import accuracy_score
training_data_accuracy = accuracy_score(y_train,x_train_pred)
print('Accuracy score of training data :',training_data_accuracy)


# In[56]:


x_test_pred = logreg.predict(x_test)


# In[57]:


training_data_accuracy = accuracy_score(y_test,x_test_pred)
print('Accuracy score of training data :',training_data_accuracy)


# # The training-set accuracy score is 0.8075 while the test-set accuracy to be 0.7821 . These two values are quite comparable .So , there is no question of overfitting

# # Metrics for model evalution
# 
# # Confusion Matrix

# In[58]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, x_test_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[59]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Negative:0', 'Actual Positive:1'], 
                                 index=['Predict Negative:0', 'Predict Positive:1'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Classification metrics 

# In[60]:


from sklearn.metrics import classification_report

print(classification_report(y_test, x_test_pred))


# In[61]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[62]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# # Classification error

# In[63]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# # Precision 

# In[64]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# # Recall

# In[65]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# # Specifity 

# In[66]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# # F1-Score

# In[70]:


# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Survived classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


# In[68]:


# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# # K fold cross validation

# In[69]:


### Applaying 5-fold cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg,x_train,y_train,cv=5,scoring='accuracy')

print('Cross-Validation scores:{}'.format(scores))


# In[ ]:




