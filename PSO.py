# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:26:59 2020

@author: user
"""

#use_polynomial = '+1.0*pow(x0,2)+1.0*pow(x1,2)'
import numpy as np
import pandas as pd
import os
def getuse_polynomial():
    now_path = os.path.join(os.getcwd(), 'use_polynomial.txt')
    f = open(now_path)
    use_polynomial=''
    for line in f:
        use_polynomial = use_polynomial+line
    return use_polynomial
###要尋找的目標方程式
def getfitness(X):
    for index,value in enumerate(X):
        exec('x'+str(index)+'='+str(value))
    use_polynomial = getuse_polynomial()
    #use_polynomial = '+1.0*pow(x1,2)' test
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
T_bestparas.to_csv('Best Parameters.csv')