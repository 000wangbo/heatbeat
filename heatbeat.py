import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from bs4 import  BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


def transform(preds):
    for i in range(len(preds)):
        if preds[i]>0.5:
            preds[i] = 1
        else:
            preds[i] = 0
    return preds

lis = []
for i in range(188):
    lis.append(str(i))


train = pd.read_csv("ptbdb_train.csv",header=None,names=lis)
test = pd.read_csv("ptbdb_test.csv",header=None,names=lis[:-1])

print(train.head(10))
print(train['187'].unique())
print(train.columns)
print(test.shape)

feature = [x for x in train.columns if x not in ['187']]

y_train = train['187']

lgb_param = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'binary',
    'metric' : {'binary_error'},
    'num_leaves' : 120,
    'learning_rate' : 0.02,
    'feature_fraction' : 0.7,
    'bagging_fraction' : 0.7,
    'bagging_freq' : 5,
    # 'min_data_in_leaf' : 10,
    'verbose' : 0
}


X_train = train[feature].values
X_test = test[feature].values

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2019)
cv_score = []
res = np.zeros(X_test.shape[0])
lgb_pred = np.zeros(X_train.shape[0])

for idx,(train_idx,test_idx) in enumerate(kf.split(X_train,y_train)):
    print('-'*50)
    print('iter{}'.format(idx+1))
    X_tr,y_tr = X_train[train_idx],y_train[train_idx]
    X_te,y_te = X_train[test_idx],y_train[test_idx]

    dtrain = lgb.Dataset(X_tr,label=y_tr)
    dvalid = lgb.Dataset(X_te, label=y_te)

    bst = lgb.train(lgb_param,dtrain,500000,valid_sets=dvalid,
                    early_stopping_rounds=150,verbose_eval=50)
    preds = bst.predict(X_te,num_iteration=bst.best_iteration)
    lgb_pred[test_idx] = preds

    preds = transform(preds)

    score = accuracy_score(y_te,preds)
    print(score)
    cv_score.append(score)
    res += transform(bst.predict(X_test,num_iteration=bst.best_iteration))

    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = train[feature].columns
    dfFeature['score'] = bst.feature_importance()
    dfFeature = dfFeature.sort_values(by='score',ascending=False)
    dfFeature.to_csv('featureImportance.csv',index=False,sep='\t')
    print(dfFeature)

print('offline mean ACC score:',np.mean(cv_score))
submission = pd.DataFrame({'id':range(len(test)),'pred':res})
submission['pred'] = submission['pred'].apply(lambda x:'0.0' if x<2.5 else '1.0')
submission['id'] = submission['id']

submission.to_csv("lgb_result.csv",index=False,header=None)




