#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats


# In[2]:


player_df = pd.read_csv("fifa19.csv")


# In[3]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']


# In[4]:


player_df = player_df[numcols+catcols]


# In[5]:


traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


# In[6]:


traindf = pd.DataFrame(traindf,columns=features)


# In[7]:


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[8]:


X.head()


# In[9]:


len(X.columns)


# ### Set some fixed set of features

# In[10]:


feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

# In[ ]:


def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    cor_list = []
    for col in X.columns:
        cor = abs(X[col].corr(y))
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_support = [True if i >= sorted(cor_list)[-num_feats] else False for i in cor_list]
    cor_feature = X.loc[:, cor_support].columns.tolist()
    # Your code ends here
    return cor_support, cor_feature


# In[15]:


cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


# ### List the selected features from Pearson Correlation

# In[16]:


cor_feature


# ## Filter Feature Selection - Chi-Sqaure

# In[17]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


# ### Chi-Squared Selector function

# In[18]:


def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)

    selector = SelectKBest(chi2, k=num_feats)
    selector.fit(X_norm, y)

    chi_support = selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    # Your code ends here
    return chi_support, chi_feature


# In[19]:


chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')


# ### List the selected features from Chi-Square 

# In[20]:


chi_feature


# ## Wrapper Feature Selection - Recursive Feature Elimination

# In[21]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# ### RFE Selector function

# In[22]:


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)

    model = LogisticRegression(max_iter=500)
    rfe = RFE(estimator=model, n_features_to_select=num_feats)
    rfe.fit(X_norm, y)

    rfe_support = rfe.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature


# In[23]:


rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


# ### List the selected features from RFE

# In[24]:


rfe_feature


# ## Embedded Selection - Lasso: SelectFromModel

# In[25]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[26]:


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)

    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
    selector = SelectFromModel(model, max_features=num_feats)
    selector.fit(X_norm, y)

    embedded_lr_support = selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature


# In[27]:


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


# In[28]:


embedded_lr_feature


# ## Tree based(Random Forest): SelectFromModel

# In[29]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[30]:


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(model, max_features=num_feats)
    selector.fit(X, y)

    embedded_rf_support = selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


# In[46]:


embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')


# In[47]:


embedded_rf_feature


# ## Tree based(Light GBM): SelectFromModel

# In[37]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


# In[38]:


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    model = LGBMClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(model, max_features=num_feats)
    selector.fit(X, y)

    embedded_lgbm_support = selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


# In[40]:


embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')


# In[41]:


embedded_lgbm_feature


# ## Putting all of it together: AutoFeatureSelector Tool

# In[49]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
# count the selected times for each feature
feature_selection_df['Total'] = feature_selection_df.iloc[:, 1:].sum(axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# In[50]:


def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    player_df = pd.read_csv(dataset_path)

    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling',
               'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
               'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance', 'ShotPower',
               'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']

    player_df = player_df[numcols + catcols]

    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    features = traindf.columns

    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf, columns=features)

    y = traindf['Overall'] >= 87
    X = traindf.copy()
    del X['Overall']

    num_feats = 30
    # Your code ends here
    return X, y, num_feats


# In[51]:


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)

    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)

    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)


    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    feature_selection_df = pd.DataFrame({'Feature': X.columns,
                                         'Pearson': cor_support, 
                                         'Chi-2': chi_support, 
                                         'RFE': rfe_support, 
                                         'Logistics': embedded_lr_support,
                                         'Random Forest': embedded_rf_support, 
                                         'LightGBM': embedded_lgbm_support})

    feature_selection_df['Total'] = feature_selection_df.iloc[:, 1:].sum(axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    best_features = feature_selection_df[
        feature_selection_df['Total'] >= len(methods) / 2
    ]['Feature'].tolist()
    #### Your Code ends here
    return best_features


# In[ ]:


best_features = autoFeatureSelector(dataset_path="fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
print(best_features)

