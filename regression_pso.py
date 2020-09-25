# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:53:11 2020

@author: user
"""
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

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:26:59 2020

@author: user
"""
print('start PSO')

#use_polynomial = '+1.0*pow(x0,2)+1.0*pow(x1,2)'
import numpy as np
import pandas as pd
import os
# def getuse_polynomial():
#     now_path = os.path.join(os.getcwd(), 'use_polynomial.txt')
#     f = open(now_path)
#     use_polynomial=''
#     for line in f:
#         use_polynomial = use_polynomial+line
#     return use_polynomial
###要尋找的目標方程式
def getfitness(X):
    for index,value in enumerate(X):
        exec('x'+str(index)+'='+str(value))
    #use_polynomial = getuse_polynomial()
    return eval(use_polynomial)

def gopso(pN=10):
    #讀取超參數最大最小值
    df = pd.read_csv('region.csv')
    Min_X = np.array(df.iloc[0,:]).astype(float)
    Max_X = np.array(df.iloc[1,:]).astype(float)
    dim = len(Min_X)
    ##設定參數
    pN=pN                        #粒子群群數
    ###兩種寫法
    #Method 1
    Max_V = np.zeros(dim,dtype='float64')
    def get_V_range(regionX,regionV,ratio):
        for param_index in range(len(Max_X)):
            regionV[param_index] = np.float64(ratio*regionX[param_index])
        return regionV
    Max_V = get_V_range(Max_X,Max_V,0.2)
    Min_V = -Max_V
    Max_Function_Call = 5       #最大函數呼叫次數
    Max_c1 = 2                  #個體學習參數 [0-4]
    Min_c1 = 0                  #個體學習參數 [0-4]
    Max_c2 = 2                  #群體學習參數 [0-4]
    Min_c2 = 0                  #群體學習參數 [0-4]
    Max_w=0.9                    #權重最大值
    Min_w=0.4                    #權重最小值
    ##產生放置空間
    X = np.zeros((pN,dim),dtype='float64')
    V = np.zeros((pN,dim),dtype='float64')
    pbest = np.zeros((pN,dim),dtype='float64')   #個體最佳位置 
    gbest = np.zeros((1,dim),dtype='float64')
    p_fit = np.zeros(pN,dtype='float64')         #自身最佳適應值 
    #Mbest = np.zeros((pN,dim),dtype='float64')   #計算新粒子位置和速度  
    #MVbest = np.zeros((pN,dim),dtype='float64')
    L = np.zeros((1),dtype='float64')            #lambda
    A = np.zeros((1),dtype='float64')            #alpha
    B = np.zeros((1),dtype='float64')            #beta
    R = np.zeros((1),dtype='float64')            #gamma
    fit = np.float64(1e50)                       #群體最佳值
    fitness = []                                 #適應值 

    #init lambda alpha beta gamma
    L[0]=1                      #分數階參數
    A[0]=1                      #w之參數
    B[0]=1                      #c1之參數 
    R[0]=1                      #c2之參數

    ####隨機初始速度與位置
    for i in range(pN):
        for j in range(dim): 
            X[i][j] = np.random.uniform(Min_X[j],Max_X[j])  
            V[i][j] = np.random.uniform(Min_V[j],Max_V[j])
        
        #由於第一代沒有比較值，所以將隨機生成的位置視為自身最佳解位置
        pbest[i] = X[i]
        ##計算初始適應值函數
        #temp = FF.function(X[i],num=No_Function)
        temp = getfitness(X[i])
        #第一代視為自身最佳值
        p_fit[i] = temp
        ##尋找全域最佳值
        if(temp < fit):
            fit = temp  
            gbest[0] = X[i]
        
    #def New_Position(X,V,pbest,gbest,BF_X1,BF_X2,BF_X3,L,A,B,R): #產生新的位置和速度
    def New_Position(X,V,pbest,gbest,L,A,B,R): #產生新的位置和速度  
        r1=np.float64(np.random.uniform(0,1))
        r2=np.float64(np.random.uniform(0,1))
        w=np.float64(Min_w + (((Max_Function_Call-Function_call_current)/Max_Function_Call)**A)*(Max_w - Min_w))
        c1=np.float64(Min_c1 + (((Max_Function_Call-Function_call_current)/Max_Function_Call)**B)*(Max_c1-Min_c1))  #B表示beta
        c2=np.float64(Max_c2 + (((Max_Function_Call-Function_call_current)/Max_Function_Call)**R)*(Min_c2-Max_c2))  #R表示gama
        '''PSO'''
        MV=w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
        '''FPSO
        MV=w*V + c1*r1*\
        (pbest - (L*X) - (L/2)*(1-L)*BF_X1 - (L/6)*(1-L)*(2-L)*BF_X2 - (L/24)*(1-L)*(2-L)*(3-L)*BF_X3)\
        + c2*r2*(gbest - X)
        '''
        #界定MV速度上下限 (最大速度法)
        
        for k in range(dim):
            if MV[k] > Max_V[k]:
                MV[k]=Max_V[k]
            if MV[k] < Min_V[k]:
                MV[k]=Min_V[k]
        M=X+MV
        #界定M位置上下限 (搜尋邊界上下限)
        for k in range(dim):
            #每個位置界限不同，因此需各自比較 Max_X->Max_X[index]
            if M[k] > Max_X[k]:
                M[k]=Max_X[k]      
            if M[k] < Min_X[k]:
                #看你對於碰到界線的處理方法而定，看哪個對你的結果比較好
                #M[k]=np.random.uniform(Min_X[k],Max_X[k])
                M[k]=Min_X[k]         
        #返回新位置與速度
        return MV,M

    #----------------------更新粒子位置----------------------------------  
    Function_call_current = 0 #初始函數呼叫次數
    while(1):
        #####停止條件 可使用迭代數或fitness達到某個停止
        if(Function_call_current >= Max_Function_Call): #當函數呼叫次數達到上限 終止演算法
            break
        Function_call_current +=1 #函數呼叫次數加一
        #####計算適應值與更新pbest和gbest
        for i in range(pN):                
            ###計算適應值
            #temp = FF.function(X[i],num=No_Function)
            temp = getfitness(X[i])
            ###更新pbest與gbest
            #更新個體最優
            if(temp<p_fit[i]):      
                p_fit[i] = temp
                pbest[i] = X[i]
                #更新全體最優
                if(p_fit[i] < fit):  
                    gbest[0] = X[i]
                    #紀錄最佳適應之值
                    fit = p_fit[i]  
        ######更新速度與位置
        for i in range(pN):
            V[i],X[i]=New_Position(X=X[i],V=V[i],pbest=pbest[i],gbest=gbest[0],L=L[0],A=A[0],B=B[0],R=R[0])
        fitness.append(fit)
    return gbest[0]
gebst = gopso()
paras=[]
paras_val=[]
for index,value in enumerate(gebst):
    paras.append('x'+str(index))
    paras_val.append(value)
bestparas = pd.DataFrame({'Parameters':paras,'value':paras_val})
T_bestparas = bestparas.T
T_bestparas.to_csv('Best Parameters.csv',header=False,index=False)

print('PSO end')