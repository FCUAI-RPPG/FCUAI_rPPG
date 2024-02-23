import pandas as pd
import numpy as np

# ////////////////////////////////////////////////////////////////////////////////////////////////////////1
input1 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗2ex4.csv",header = None).transpose()
ans1 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗2ex4_bpm.csv",header = None)

median = input1.iloc[:,-1]
bpm_true = ans1
input1.drop(input1.columns[[0,-1]], axis=1, inplace=True)
input1 = pd.concat([input1,ans1,median], axis=1)

train1 = input1.iloc[0:int(len(input1) * 0.7),:]
test1 = input1.drop(train1.index)
test1.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test1.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////2
input2 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗3_4.csv",header = None).transpose()
ans2 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗3_4_bpm.csv",header = None)

median = input2.iloc[:,-1]
bpm_true = ans2
input2.drop(input2.columns[[0,-1]], axis=1, inplace=True)
input2 = pd.concat([input2,ans2,median], axis=1)

train2 = input2.iloc[0:int(len(input2) * 0.7),:]
test2 = input2.drop(train2.index)
test2.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test2.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////3
input3 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗3_5.csv",header = None).transpose()
ans3 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗3_5_bpm.csv",header = None)

median = input3.iloc[:,-1]
bpm_true = ans3
input3.drop(input3.columns[[0,-1]], axis=1, inplace=True)
input3 = pd.concat([input3,ans3,median], axis=1)

train3 = input3.iloc[0:int(len(input3) * 0.7),:]
test3 = input3.drop(train3.index)
test3.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test3.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////4
input4 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗3_6.csv",header = None).transpose()
ans4 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗3_6_bpm.csv",header = None)

median = input4.iloc[:,-1]
bpm_true = ans4
input4.drop(input4.columns[[0,-1]], axis=1, inplace=True)
input4 = pd.concat([input4,ans4,median], axis=1)

train4 = input4.iloc[0:int(len(input4) * 0.7),:]
test4 = input4.drop(train4.index)
test4.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test4.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////5
input5 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗4ex3_發呆1.csv",header = None).transpose()
ans5 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗4ex3_發呆1_bpm.csv",header = None)

median = input5.iloc[:,-1]
bpm_true = ans5
input5.drop(input5.columns[[0,-1]], axis=1, inplace=True)
input5 = pd.concat([input5,ans5,median], axis=1)

train5 = input5.iloc[0:int(len(input5) * 0.7),:]
test5 = input5.drop(train5.index)
test5.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test5.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////6
input6 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗4ex3影片1.csv",header = None).transpose()
ans6 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗4ex3影片1_bpm.csv",header = None)

median = input6.iloc[:,-1]
bpm_true = ans6
input6.drop(input6.columns[[0,-1]], axis=1, inplace=True)
input6 = pd.concat([input6,ans6,median], axis=1)

train6 = input6.iloc[0:int(len(input6) * 0.7),:]
test6 = input6.drop(train6.index)
test6.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test6.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////7
input7 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗4ex3影片2.csv",header = None).transpose()
ans7 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗4ex3影片2_bpm.csv",header = None)

median = input7.iloc[:,-1]
bpm_true = ans7
input7.drop(input7.columns[[0,-1]], axis=1, inplace=True)
input7 = pd.concat([input7,ans7,median], axis=1)

train7 = input7.iloc[0:int(len(input7) * 0.7),:]
test7 = input7.drop(train7.index)

test7.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test7.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////8
input8 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗6ex2發呆.csv",header = None).transpose()
ans8 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗6ex2發呆_bpm.csv",header = None)

median = input8.iloc[:,-1]
bpm_true = ans8
input8.drop(input8.columns[[0,-1]], axis=1, inplace=True)
input8 = pd.concat([input8,ans8,median], axis=1)

train8 = input8.iloc[0:int(len(input8) * 0.7),:]
test8 = input8.drop(train8.index)
test8.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test8.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////9
input9 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗6ex2影片.csv",header = None).transpose()
ans9 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗6ex2影片_bpm.csv",header = None)

median = input9.iloc[:,-1]
bpm_true = ans9
input9.drop(input9.columns[[0,-1]], axis=1, inplace=True)
input9 = pd.concat([input9,ans9,median], axis=1)

train9 = input9.iloc[0:int(len(input9) * 0.7),:]
test9 = input9.drop(train9.index)
test9.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test9.csv',index = False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////10
input10 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗9_ex3影片.csv",header = None).transpose()
ans10 = pd.read_csv("D://rPPG//測試每個patch算出的數據//實驗9_ex3影片_bpm.csv",header = None)

median = input10.iloc[:,-1]

bpm_true = ans10
input10.drop(input10.columns[[0,-1]], axis=1, inplace=True)
input10 = pd.concat([input10,ans10,median], axis=1)

train10 = input10.iloc[0:int(len(input10) * 0.7),:]
test10 = input10.drop(train10.index)
test10.to_csv('D://rPPG//測試每個patch算出的數據//train_test//test10.csv',index = False)


pd.concat([train1,train2,train3,train4,train5,train6,train7,train8,train9,train10],axis =0).to_csv('D://rPPG//測試每個patch算出的數據//train_test//train.csv',index = False)
