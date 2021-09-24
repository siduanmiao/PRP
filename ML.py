from sklearn.linear_model import Lasso
import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate

list_a=np.arange(40)
tmp = list(list_a)

#去除掉三列无用信息
tmp.remove(1)
tmp.remove(2)
tmp.remove(6)



trail = pd.read_csv('data_trail_set.csv',index_col=0,usecols=tmp)
valid = pd.read_csv('vaidate_set.csv',index_col=0,usecols=tmp)
trail_Y_p = trail.pop("DrugOrNonDrug")
trail_X_p = trail
valid_Y_p = valid.pop("DrugOrNonDrug")
valid_X_p = valid

trail_X, trail_Y = sklearn.utils.shuffle(trail_X_p, trail_Y_p, random_state=1)
valid_X, valid_Y = sklearn.utils.shuffle(valid_X_p, valid_Y_p, random_state=1)

lasso = Lasso(alpha=0.001, max_iter=100000).fit(trail_X, trail_Y)
name=[]

for i in lasso.coef_:
    if abs(i) > 0.0 :
        name = name + [1]
    else :
        name = name + [0]
print("我们选择的特征是：")
print(name)

model = SelectFromModel(lasso,prefit=True)

t_X=model.transform(trail_X)
t_Y=trail_Y
v_X=model.transform(valid_X)
v_Y=valid_Y




'''
#使用迭代特征选择(RFE)进行特征筛选。
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),n_features_to_select=15)
select.fit(trail_X, trail_Y)
print("所选择出来的15个特征：")
print(select.get_support())









t_X=select.transform(trail_X)
t_Y=trail_Y
v_X=select.transform(valid_X)
v_Y=valid_Y
'''





# 我们假设最终选取出来的特征集为t_X,t_Y,v_X,v_Y
# 我们使用带交叉验证的网格搜索来获得最优的参数,由于数量级相差较大，我们使用MinMaxScaler进行数据放缩
# 为了防止正则化的过程中信息泄露并简化对多个模型的操作方法，我们使用管道

#print(v_X['DG_P'])

print("朴素贝叶斯:")
print("The train set :")
pipe = Pipeline([("scaler", MinMaxScaler()), ("NB", GaussianNB())])
grid = cross_val_score(pipe,t_X,t_Y,cv=5)
print("Best cross-validation accuracy: {:.2f}".format(grid.mean()))
scoring = ['precision_macro', 'recall_macro','f1_macro', 'roc_auc']
scores = cross_validate(pipe, t_X, t_Y, scoring=scoring,cv=5)

for i in scoring :

    print("%s:%.2f"%(i,np.mean(scores['test_'+i])))








change=MinMaxScaler()
change.fit(t_X, t_Y)
gs=GaussianNB()
gs.fit(change.transform(t_X), t_Y)
score=gs.score(change.transform(v_X),v_Y)
print("Test set score: {:.3f}".format(gs.score(change.transform(v_X), v_Y)))
pred_logreg = gs.predict(change.transform(v_X))
confusion = confusion_matrix(v_Y, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(v_Y, pred_logreg))
try:
    auc = roc_auc_score(v_Y, gs.predict_proba(change.transform(v_X))[:, 1])
    print("AUC: {:.3f}".format(auc))
except AttributeError:
    auc = roc_auc_score(v_Y, gs.decision_function(change.transform(v_X)))
    print("AUC: {:.3f}".format(auc))



# 核SVC
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 20,30,40,50,60,70,80,90,100],'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(t_X,t_Y)
print("核SVM：")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best parameters: {}".format(grid.best_params_))
pipe.set_params(**grid.best_params_)
scoring = ['precision_macro', 'recall_macro','f1_macro', 'roc_auc']
scores = cross_validate(pipe, t_X, t_Y, scoring=scoring,cv=5)

for i in scoring :

    print("%s:%.2f"%(i,np.mean(scores['test_'+i])))
pipe.fit(t_X,t_Y)
print("Test set score: {:.3f}".format(pipe.score(v_X,v_Y)))
pred_logreg=pipe.predict(v_X)
confusion = confusion_matrix(v_Y, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(v_Y, pred_logreg))
try:
    auc = roc_auc_score(v_Y, pipe.predict_proba(v_X)[:, 1])
    print("AUC: {:.3f}".format(auc))
except AttributeError:
    auc = roc_auc_score(v_Y, pipe.decision_function(v_X))
    print("AUC: {:.3f}".format(auc))






# Logistic
param_grid = {'lg__C': [0.001, 0.01, 0.1, 1, 10, 100,1000,10000],'lg__max_iter':[100000]}
pipe = Pipeline([("scaler", MinMaxScaler()), ("lg", LogisticRegression())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(t_X,t_Y)
print("Logistic：")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best parameters: {}".format(grid.best_params_))
pipe.set_params(**grid.best_params_)
scoring = ['precision_macro', 'recall_macro','f1_macro', 'roc_auc']
scores = cross_validate(pipe, t_X, t_Y, scoring=scoring,cv=5)

for i in scoring :

    print("%s:%.2f"%(i,np.mean(scores['test_'+i])))
pipe.fit(t_X,t_Y)
print("Test set score: {:.3f}".format(pipe.score(v_X,v_Y)))
pred_logreg=pipe.predict(v_X)
confusion = confusion_matrix(v_Y, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(v_Y, pred_logreg))
try:
    auc = roc_auc_score(v_Y, pipe.predict_proba(v_X)[:, 1])
    print("AUC: {:.3f}".format(auc))
except AttributeError:
    auc = roc_auc_score(v_Y, pipe.decision_function(v_X))
    print("AUC: {:.3f}".format(auc))






#KNN
param_grid = {'knn__n_neighbors': [1,2,3,4,5,6,7,8,9]}
pipe = Pipeline([("scaler", MinMaxScaler()), ("knn", KNeighborsClassifier())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(t_X,t_Y)
print("KNN：")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best parameters: {}".format(grid.best_params_))
pipe.set_params(**grid.best_params_)
scoring = ['precision_macro', 'recall_macro','f1_macro', 'roc_auc']
scores = cross_validate(pipe, t_X, t_Y, scoring=scoring,cv=5)

for i in scoring :

    print("%s:%.2f"%(i,np.mean(scores['test_'+i])))
pipe.fit(t_X,t_Y)
print("Test set score: {:.3f}".format(pipe.score(v_X,v_Y)))
pred_logreg=pipe.predict(v_X)
confusion = confusion_matrix(v_Y, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(v_Y, pred_logreg))
try:
    auc = roc_auc_score(v_Y, pipe.predict_proba(v_X)[:, 1])
    print("AUC: {:.3f}".format(auc))
except AttributeError:
    auc = roc_auc_score(v_Y, pipe.decision_function(v_X))
    print("AUC: {:.3f}".format(auc))





#LinearSVC
param_grid = {'Lsvc__C': [0.001, 0.01, 0.1, 1, 10, 100,1000],'Lsvc__max_iter':[1000000]}
pipe = Pipeline([("scaler", MinMaxScaler()), ("Lsvc", LinearSVC())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(t_X,t_Y)
print("LinearSVC:")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best parameters: {}".format(grid.best_params_))
pipe.set_params(**grid.best_params_)
scoring = ['precision_macro', 'recall_macro','f1_macro', 'roc_auc']
scores = cross_validate(pipe, t_X, t_Y, scoring=scoring,cv=5)

for i in scoring :

    print("%s:%.2f"%(i,np.mean(scores['test_'+i])))
pipe.fit(t_X,t_Y)
print("Test set score: {:.3f}".format(pipe.score(v_X,v_Y)))
pred_logreg=pipe.predict(v_X)
confusion = confusion_matrix(v_Y, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(v_Y, pred_logreg))
try:
    auc = roc_auc_score(v_Y, pipe.predict_proba(v_X)[:, 1])
    print("AUC: {:.3f}".format(auc))
except AttributeError:
    auc = roc_auc_score(v_Y, pipe.decision_function(v_X))
    print("AUC: {:.3f}".format(auc))





# 随机森林
param_grid = {'RF__n_estimators': [1,10,50,100,150,200,250],'RF__max_features': [1,2,3,4,5,6,7,8,9,10,11,12]}
pipe = Pipeline([("scaler", MinMaxScaler()), ("RF", RandomForestClassifier())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(t_X,t_Y)
print("随机森林：")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best parameters: {}".format(grid.best_params_))
pipe.set_params(**grid.best_params_)
scoring = ['precision_macro', 'recall_macro','f1_macro', 'roc_auc']
scores = cross_validate(pipe, t_X, t_Y, scoring=scoring,cv=5)

for i in scoring :

    print("%s:%.2f"%(i,np.mean(scores['test_'+i])))
pipe.fit(t_X,t_Y)
print("Test set score: {:.3f}".format(pipe.score(v_X,v_Y)))
pred_logreg=pipe.predict(v_X)
confusion = confusion_matrix(v_Y, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(v_Y, pred_logreg))
try:
    auc = roc_auc_score(v_Y, pipe.predict_proba(v_X)[:, 1])
    print("AUC: {:.3f}".format(auc))
except AttributeError:
    auc = roc_auc_score(v_Y, pipe.decision_function(v_X))
    print("AUC: {:.3f}".format(auc))

# xgboost
param_grid = {'xgb__n_estimators': [1,10,50,100,150,200,250],'xgb__max_depth': [1,2,3,4,5,6,7,8,9,10],'xgb__learning_rate':[0.01,0.02,0.03,0.04,0.1,0.15,0.2,0.25,0.3],'xgb__eval_metric':[['logloss','auc','error']]}
pipe = Pipeline([("scaler", MinMaxScaler()), ("xgb", XGBClassifier())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(t_X,t_Y)
print("xgboost：")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best parameters: {}".format(grid.best_params_))
pipe.set_params(**grid.best_params_)
scoring = ['precision_macro', 'recall_macro','f1_macro', 'roc_auc']
scores = cross_validate(pipe, t_X, t_Y, scoring=scoring,cv=5)

for i in scoring :

    print("%s:%.2f"%(i,np.mean(scores['test_'+i])))
pipe.fit(t_X,t_Y)
print("Test set score: {:.3f}".format(pipe.score(v_X,v_Y)))
pred_logreg=pipe.predict(v_X)
confusion = confusion_matrix(v_Y, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(v_Y, pred_logreg))
try:
    auc = roc_auc_score(v_Y, pipe.predict_proba(v_X)[:, 1])
    print("AUC: {:.3f}".format(auc))
except AttributeError:
    auc = roc_auc_score(v_Y, pipe.decision_function(v_X))
    print("AUC: {:.3f}".format(auc))

