# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:42:09 2020

@author: user
"""
#畫圖沒法打包
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import os

##讀取csv路徑
def get_csv(filename):
    df = pd.read_csv(filename+'.csv')
    return df

#訓練資料
def get_trainingdata(df):   
    df_colums = df.columns
    Y = df[df_colums[-1]] #Y=df['fitness']
    X = df.drop(df_colums[-1],axis = 1)
    return X,Y,df_colums
#顯示多項式
def get_show_polynomial(coef,featur_name):
    polynomial = ''
    for value,fea in zip(coef,featur_name):
        fea = fea.replace(' ', '')
        value = round(value,4)
        if value ==0:
            continue
        if fea =='1':
            if value > 0:
                polynomial=polynomial+str(value)
            if value < 0:
                polynomial=polynomial+str(value)
        else:
            if value > 0:
                polynomial=polynomial+'+'+str(value)+fea
            if value < 0:
                polynomial=polynomial+str(value)+fea
    return polynomial
#將敘述式變成表達式
def get_polynomial(coef,featur_name):
    polynomial = ''
    for value,fea in zip(coef,featur_name):
        fea = fea.replace(' ', '*')
        value = round(value,4)
        if value ==0:
            continue
        if fea =='1':
            if value > 0:
                polynomial=polynomial+str(value)
            if value < 0:
                polynomial=polynomial+str(value)
        else:
            #假如有次方項
            if '^' in fea:
                ss = fea.split('^')
                fea = 'pow('+ss[0]+','+ss[1]+')'
            if value > 0:
                polynomial=polynomial+'+'+str(value)+'*'+fea
            if value < 0:
                polynomial=polynomial+str(value)+'*'+fea
    return polynomial

##創建資料夾模板
def mkdir(path):
    #判斷目錄是否存在
    #存在：True
    #不存在：False
    folder = os.path.exists(path)
    #判斷結果
    if not folder:
        #如果不存在，則建立新目錄
        os.makedirs(path)
    else:
        #如果目錄已存在，則不建立
        pass

#相關係數
def plt_R(true,predict,xname='Y_true',yname='Y_pred',save_flag = False):
    Rsquared = np.corrcoef(true, predict)[0,1]
    Rsquared = round(Rsquared,4)
    #plt.figure(figsize=(6,6))
    #plt.scatter(true,predict,label='R='+str(Rsquared))
    #plt.title('correlation')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    #plt.legend()
    #print(f'R^2:{Rsquared}')
    if save_flag == True:
        #儲存路徑
        now_path = os.getcwd()
        save_folder = 'save_folder'
        #save_path = os.path.join(now_path,save_folder)
        mkdir(save_folder)
        #fig_name = "correlation.png"
        #plt.savefig(os.path.join(save_folder,fig_name))
    return Rsquared

def doall():
    print('start regression')
    filename = 'expdata'
    df = get_csv(filename)
    X,Y,df_colums = get_trainingdata(df)
    #多項式回歸，可調參數為degree
    polydegree = 2
    mutipoly_leg = Pipeline([('poly', PolynomialFeatures(degree=polydegree)),
                      ('linear', LinearRegression(fit_intercept=False))])
    mutipoly_leg = mutipoly_leg.fit(X,Y)
    predict_y = mutipoly_leg.predict(X)
    coef = mutipoly_leg.named_steps['linear'].coef_
    featur_name = (mutipoly_leg.named_steps['poly'].get_feature_names())
    show_polynomial = get_show_polynomial(coef,featur_name)#字串
    with open("show_polynomial.txt","w") as f:
        f.write(show_polynomial)
    use_polynomial = get_polynomial(coef,featur_name)
    with open("use_polynomial.txt","w") as f:
        f.write(use_polynomial)
    #print(featur_name)
    #print(coef)
    R = plt_R(Y, predict_y,xname='Y_true',yname='Y_pred',save_flag=False)
    with open("R.txt","w") as f:
        f.write(str(R))
    print('regression end')
    
    
    return use_polynomial
    
use_polynomial = doall()