#!/usr/bin/env python
# coding: utf-8

# In[96]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[97]:


import os
import pandas as pd
import numpy as np


# In[98]:


os.chdir("C:/Users/utsav/Desktop/Ayushi/Python/Project files")


# In[99]:


data = pd.read_csv("cardio_train.csv", sep=';')


# In[100]:


data.head()
#age is in days


# In[101]:


data.columns


# In[102]:


# Verify whether any null values present in the dataset
data.isnull().sum()


# In[103]:


# Check datatype
data.info() 


# In[104]:


#Descriptive Statistics on dataset
data.describe()


# In[105]:


from matplotlib import rcParams


# In[106]:


#Age is measured in days, height is in centimeters. 
#Let's look ate the numerical variables and how are they spread among target class. 
#For example, at what age does the number of people with CVD exceed the number of people without CVD?
rcParams['figure.figsize'] = 11, 8
data['years'] = (data['age']/365).round().astype('int')
import seaborn as sns
sns.countplot(x = 'years', hue = 'cardio', data = data)

# we can see that from age of 55, the chances of having cvd increases.


# In[108]:


## If we plot simple distplot of random numbers, we'll get normal distribution
x = np.random.randn(100)
norm_dist = sns.distplot(x)


# In[109]:


sns.distplot(data['years'] )


# In[110]:


sns.boxplot(x = 'cardio', y = 'years', data = data)
## We can see that people with cardio disease are older than people with no cardio disease


# In[111]:


#Let's look at categorical variables in the dataset and their distribution:
data_categorical = data.loc[:,['cholesterol','gluc','smoke','active','alco']]
sns.countplot(x="variable", hue="value",data= pd.melt(data_categorical));


# In[112]:


pd.melt(data_categorical)


# In[113]:


#Bivariate analysis
#It may be useful to split categorical variables by target class
data_long = pd.melt(data, id_vars=['cardio'],value_vars=['cholesterol','gluc','smoke','active','alco'])


# In[114]:


data_long


# In[115]:


import seaborn as sns
sns.catplot(x = 'variable', hue = 'value', col = 'cardio', data= data_long, kind='count')
# You can see that people with CVD have higher glucose and cholesterol level(2,3), but smoke, activeness and alcohol level doesn't show us much difference.


# In[116]:


## In our data, the gender column has 1 and 2 values but we don't know which value represents male and female.
## So we'll calculate average height for both. And we'll assume that males are taller than females.

data['gender'].value_counts()


# In[117]:


sns.countplot(x = 'cardio', hue = 'gender', data = data, palette='Set2')
## gender vs cardio.. you can see the number of males and females for each value of cardio


# In[118]:


data['weight'].describe()


# In[119]:


sns.boxplot(x = 'cardio', y = 'weight', data = data)


# In[120]:


data.groupby('gender')['height'].mean()
# so males are 2 and females are 1


# In[121]:


## Lets see who consumes more alcohol

data.groupby('gender')['alco'].sum()

## We can see that men consume more alcohol than females


# In[122]:


## Let's see if the target variables is balanced or not.

data['cardio'].value_counts(normalize = True)

## Balanced data set because both values 1 and 0 are equally distributed in the dataset.


# In[123]:


## Pandas Crosstab function to see if the dataset is balanced or not
pd.crosstab(data['gender'],data['cardio'])  ## use normalize to see the percentage of values


# In[124]:


## Checking for correlations of all attibutes with the target variable

data.corr()['cardio'].drop('cardio')


# In[125]:


data.hist(figsize=(15,20))


# In[130]:


## Renaming columns
data.columns = ['id','age','gender', 'height', 'weight', 'Systolic_BP', 'Diastolic_BP',
      'cholesterol', 'glucose', 'smoke', 'alcohol', 'active', 'cardio', 'years']
#data.columns = ['id','gender', 'height', 'weight', 'ap_hi', 'ap_lo',
      # 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'years']


# In[131]:


data.head()


# In[10]:


## Outliers check using BoxPlot


# In[134]:


import matplotlib.pyplot as plt
columns = [ 'gender', 'height', 'weight', 'Systolic_BP', 'Diastolic_BP',
       'cholesterol', 'glucose', 'smoke', 'alcohol', 'active', 'cardio', 'years']
for i in range(len(columns)):
    def check_outliers(i):
        fig,axes=plt.subplots(1,1)
        sns.boxplot(data=data,x=i, color='Green')
        fig.set_size_inches(15,5)


# In[135]:


for i in range(len(columns)):
    check_outliers(columns[i])
    
# We can see that height, weight, Systolic BP, Diastolic BP have outliers. Lets remove them.
# Also, year has one outlier.
# Also, we can see that, there are negative values of Systolic and Dialostic BP which doesn't make any sense. 
# And, for some records, Systolic BP is less than Dialostic BP which is inappropriate. So we'll get rid of them too.


# In[136]:


from sklearn.model_selection import train_test_split

X = data.iloc[:,0:11]
y = data.iloc[:,12]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[137]:


from sklearn.linear_model import LogisticRegression


# In[138]:


logreg = LogisticRegression().fit(X_train,y_train)



# print the coefficients and model performance
print("Logistic Intercept:", logreg.intercept_)
print("training Score:{:.3f}".format(logreg.score(X_train, y_train)) )
print("testing Score:{:.3f}".format(logreg.score(X_test, y_test)) )


# In[ ]:


## Lets see results after we remove outliers


# In[139]:


## Lets remove the outliers form the mentioned columns: height,weight,systolic BP, Diastolic BP
print(data.height.describe())
data[data['height'] > 200]


# In[140]:


del data['age']


# In[141]:


ntive_BP_rows = data[data['Systolic_BP']<0]## we'll delete these records cuz BP can't be negative
ntive_BP_rows


# In[142]:


neg_dias_bp = data[data['Diastolic_BP']<0] # we'll delete this record because BP can't be negative
neg_dias_bp


# In[143]:


data = data[~(data['Systolic_BP']<0)]


# In[144]:


data = data[~(data['Diastolic_BP']<0)]


# In[145]:


data[data.Systolic_BP < 0] ## no more negative values of Systolic BP


# In[146]:


data[data.Diastolic_BP<0] # no more negative values of Diastolic BP


# In[147]:


data.shape


# In[148]:


## Lets remove height and weight values which are outliers

columns = ['height','weight']
for i in range(len(columns)):
    data.drop(data[(data[columns[i]] > data[columns[i]].quantile(0.975)) | (data[columns[i]] < data[columns[i]].quantile(0.025))].index,inplace=True)


# In[149]:


# Lets plot height and weight box plot again to check outliers

import seaborn as sns
sns.boxplot(x = 'cardio', y = 'weight', data = data)


# In[150]:


sns.boxplot(x = 'cardio', y = 'height', data = data)


# In[151]:


data.head()


# In[152]:


## Rearranging the columns to avoid wrong splitting
data = data[['years','gender', 'height', 'weight', 'Systolic_BP', 'Diastolic_BP',
       'cholesterol', 'glucose', 'smoke', 'alcohol', 'active', 'cardio']]

data.head()


# In[153]:


from sklearn.model_selection import train_test_split

X = data.iloc[:,0:10]
y = data.iloc[:,11]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[154]:


X.head()


# In[155]:


y.head()


# In[156]:


## Now that we have cleaned our data little bit, lets check for accuracy again.



logreg = LogisticRegression().fit(X_train,y_train)



# print the coefficients and model performance
print("Logistic Intercept:", logreg.intercept_)
print("training Score:{:.3f}".format(logreg.score(X_train, y_train)) )
print("testing Score:{:.3f}".format(logreg.score(X_test, y_test)) )


# In[157]:


# Lets try using Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# In[158]:


# Lets try using Random Forest Algorithm

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred = rnd_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))


# In[175]:


## AdaBoost Algorithm

from sklearn.ensemble import AdaBoostClassifier

ada_clf_DT = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=4), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf_DT.fit(X_train, y_train)


# In[160]:


y_pred = ada_clf_DT.predict(X_test)

print(accuracy_score(y_test, y_pred))


# In[161]:


## Lets do feature engineering- adding a new column BMI= mass(kg)/height^2(meters)

data['BMI'] = data['weight']/(data['height']/100)**2


# In[162]:


data.head()


# In[163]:


#Rearranging columns again

data = data[['years','gender', 'height', 'weight', 'BMI','Systolic_BP', 'Diastolic_BP',
       'cholesterol', 'glucose', 'smoke', 'alcohol', 'active', 'cardio']]


# In[164]:


from sklearn.model_selection import train_test_split

X = data.iloc[:,0:10]
y = data.iloc[:,11]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[165]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train,y_train)



# print the coefficients and model performance
print("Logistic Intercept:", logreg.intercept_)
print("training Score:{:.3f}".format(logreg.score(X_train, y_train)) )
print("testing Score:{:.3f}".format(logreg.score(X_test, y_test)) )


# In[166]:


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred = rnd_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))


# In[167]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# In[168]:


data.head()


# In[169]:


## New column Obesity.. 
data['Obesity'] = 0


# In[170]:


data['Obesity'] = np.where(data['cholesterol']>2, 1, 0)
        


# In[171]:


data.head()


# In[172]:


## Rearranging columns

data = data[['years','gender', 'height', 'weight', 'BMI','Systolic_BP', 'Diastolic_BP',
       'cholesterol', 'glucose', 'Obesity','smoke', 'alcohol', 'active', 'cardio']]


# In[173]:


logreg = LogisticRegression().fit(X_train,y_train)



# print the coefficients and model performance
print("Logistic Intercept:", logreg.intercept_)
print("training Score:{:.3f}".format(logreg.score(X_train, y_train)) )
print("testing Score:{:.3f}".format(logreg.score(X_test, y_test)) )

