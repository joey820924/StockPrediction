#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as Sdf
from svmutil import *
import os
os.chdir('/Users/joey/libsvm-3.22/tools/')

#input_file = /Users/joey/Documents/stockcal/
#PPP = /Users/joey/Documents/stockcal/cal/
#output_file = /Users/joey/Documents/stockcal/libsvm_format/
#rulepath = "/Users/joey/Documents/stockcal/rule/"
#pscalepath = "/Users/joey/Documents/stockcal/scale/"
#modelpath = "/Users/joey/Documents/stockcal/model/"
#gridpath = "/Users/joey/Documents/stockcal/grid/"

df = pd.read_csv("/Users/joey/Documents/stock.csv",parse_dates=True)

#指標
indexName = ['macd','macds','macdh','kdjk','kdjd','kdjj','rsi_5','rsi_10']

#獲得每支股票代號
code = df['stockcode'].unique()
'''
#將各股票轉換至各別csv

for i in code:
    tmp = df[df['stockcode'] == i] 
    apath = path + str(i)
    tmp = tmp[['close','date','amount','high','low','open','volume']]
    tmp = tmp.fillna(method='bfill')
    tmp = tmp.fillna(method = 'ffill')
    tmp = tmp.where(pd.notnull(tmp), tmp.mean(), axis='columns')
    tmp.to_csv(apath,index=False)
'''
#計算指標
input_file = []
output_file = []
path2 = '/Users/joey/Documents/stockcal/libsvm_format/'
path1 = '/Users/joey/Documents/stockcal/'

for i in code:
    ipath = path1+str(i)
    opath = path2 + str(i)+'.txt'
    input_file.append(ipath)
    output_file.append(opath)
ppp = "/Users/joey/Documents/stockcal/cal/"
PPP = []
for i in code:
    PPP.append(ppp+str(i))
'''
for i in range(len(input_file)):
    sc = Sdf.retype(pd.read_csv(input_file[i]))
    tmp = pd.DataFrame()
    tmp['target'] = sc['close'].shift(1)
    for j in indexName:
        tmp[j] = sc.get(j)
    tmp = tmp.fillna(method = 'bfill')
    tmp = tmp.fillna(method = 'ffill')    
    tmp = tmp.where(pd.notnull(tmp), tmp.mean(), axis='columns')
    tmp.to_csv(PPP[i],index=False)

#轉換成Libsvm格式
import sys
import csv
from collections import defaultdict

def construct_line( label, line ):
    new_line = []
    if float( label ) == 0.0:
        label = "0"
    new_line.append( str(float(label) ))

    for i, item in enumerate( line ):
        new_item = "%s:%s" % ( i + 1, item )
        new_line.append( new_item )
    new_line = " ".join( new_line )
    new_line += "\n"
    return new_line

# ---

def convert(input_file,output_file):
    label_index = 0

    i = open( input_file, 'rb' )
    o = open( output_file, 'wb' )

    reader = csv.reader( i )
    headers = reader.next()

    for line in reader:
        label = line.pop( label_index )

        new_line = construct_line( label, line )
        o.write( new_line )

#轉換開始
for i in range(len(PPP)):
    convert(PPP[i],output_file[i])
'''
#設置檔案路徑
prule = "/Users/joey/Documents/stockcal/rule/"
pscale = "/Users/joey/Documents/stockcal/scale/"
pmodel = "/Users/joey/Documents/stockcal/model/"
pgrid = "/Users/joey/Documents/stockcal/grid/"
rulepath = []
scalepath = []
gridpath = []
modelpath = []
for i in code:
    rulepath.append(prule+str(i)+".txt")
    scalepath.append(pscale+str(i)+".txt")
    gridpath.append(pgrid+str(i)+".txt")
    modelpath.append(pmodel+str(i)+".txt")
'''
#Scale
for i in range(len(rulepath)):
    params = "svm-scale -s "+rulepath[i]+" "+output_file[i]+" >"+scalepath[i]
    os.system(params)
'''
#Grid
'''
for i in range(len(gridpath)):
        try:
            params = "python grid2.py -log2c -5,5,1 -log2g -5,5,1 -log2p -5,5,1 -out "+gridpath[i]+" -s 3 -t 2 "+scalepath[i]
            os.system(params)
        except:
            print scalepath[i]
        continue
'''
#Train
def trainModel(grid,model,data,ori):
    origin = pd.read_csv(ori)
    lens = origin.shape[0]
    trainSize = int(0.8*lens)
    y,x = svm_read_problem(data)
    
    
    rule = pd.read_table(grid,sep = " ",header = None)
    log2c = []
    log2g = []
    log2p = []
    mse = []
    for i in rule.ix[:,0].values:
        log2c.append(i.strip("log2c="))
    for i in rule.ix[:,1].values:
        log2g.append(i.strip("log2g="))
    for i in rule.ix[:,2].values:
        log2p.append(i.strip("log2p="))
    for i in rule.ix[:,3].values:
        mse.append(i.strip("mse="))
    rule.ix[:,0] = pd.Series(log2c)
    rule.ix[:,1] = pd.Series(log2g)
    rule.ix[:,2] = pd.Series(log2p)
    rule.ix[:,3] = pd.Series(mse)
    rule.iloc[:,0] = rule.iloc[:,0].apply(lambda x:int(float(x)))
    rule.iloc[:,1] = rule.iloc[:,1].apply(lambda x:int(float(x)))
    rule.iloc[:,2] = rule.iloc[:,2].apply(lambda x:int(float(x)))
    rule = rule[rule.ix[:,3] == rule.ix[:,3].min()].iloc[-1,:]
    rule.iloc[0] = 2**int(rule.iloc[0])
    rule.iloc[1] = 2**int(rule.iloc[1])
    rule.iloc[2] = 2**int(rule.iloc[2])
    params1 = '-c '+str(rule.iloc[0])+" -s 3 -t 2 -g "+str(rule.iloc[1])+" -p "+str(rule.iloc[2])
    #params1 = "-c 1 -s 3 -t 2 -p 0.03125 -g 0.03125"
    m = svm_train(y[:trainSize],x[:trainSize],params1)
    svm_save_model(model,m)

#Train Start
#for i in range(len(scalepath)):
    #trainModel(gridpath[i],modelpath[i],scalepath[i],input_file[i])

def getAccuracy(code):
    MP = "/Users/joey/Documents/stockcal/model/"+str(code)+".txt"
    m = svm_load_model(MP)
    path = '/Users/joey/Documents/stockcal/'+str(code)
    origin = pd.read_csv(path)
    lens = origin.shape[0]
    trainSize = int(0.8*lens)
    data = "/Users/joey/Documents/stockcal/libsvm_format/"+str(code)+".txt"
    y,x = svm_read_problem(data)
    p_label, p_acc, p_val = svm_predict(y[trainSize:], x[trainSize:], m)
    #result = "name: %s , Mean squared error: %f , Squared correlation %f\n"%(str(code),p_acc[1],p_acc[2])
    predict = pd.Series(p_label)
    predict = predict-predict.shift(1)
    predict = predict.dropna(axis = 0)
    predict = predict.apply(lambda x:1 if x>0 else 0)
    true = pd.Series(y[trainSize:])
    true = true - true.shift(1)
    true = true.dropna(axis = 0)
    true = true.apply(lambda x:1 if x>0 else 0)
    a = true == predict
    b = 0
    a = a.apply(lambda x:1 if x==True else 0)
    for i in a:
        b +=i
    if float(b) != 0.0 or float(a.shape[0])!= 0.0:
        accuracy = float(b)/float(a.shape[0])
    else: accuracy = 0
    return accuracy

def getPlot(code):
    MP = "/Users/joey/Documents/stockcal/model/"+str(code)+".txt"
    m = svm_load_model(MP)
    
    data = "/Users/joey/Documents/stockcal/libsvm_format/"+str(code)+".txt"
    path = '/Users/joey/Documents/stockcal/'+str(code)
    origin = pd.read_csv(path)
    lens = origin.shape[0]
    plt.figure(figsize=(10,8)) 
    trainSize = int(0.8*lens)
    testSize = lens - trainSize
    y,x = svm_read_problem(data)
    p_label, p_acc, p_val = svm_predict(y[trainSize:], x[trainSize:], m)
    with plt.style.context('fivethirtyeight'):
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
        plt.plot(np.arange(testSize), y[trainSize:], color = 'blue', marker = 'o',linewidth=2,
                label = 'True')
        plt.plot(np.arange(testSize), p_label, color = 'black',linewidth=2, 
                marker = '*', label = 'Predict')
        #生成图例并指定图例位置。本例中loc='best'和loc='upper left'效果相同
        plt.legend(loc = 'best')
        plt.title(str(code))
        plt.show()

def getPrice(code,data):
    MP = "/Users/joey/Documents/stockcal/model/"+str(code)+".txt"
    m = svm_load_model(MP)
    y,x = svm_read_problem(data)
    p_label, p_acc, p_val = svm_predict(y, x, m)
    return p_label
