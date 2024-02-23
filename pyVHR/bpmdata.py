import pandas as pd
import numpy as np
num = "5-exercise"
input = pd.read_csv("D://訓練影片//GO//FCUAI_rppgdata//ex9//{}_bpm.txt".format(num),names=['Data'])
input.drop([0], axis=0, inplace=True)

new1 = input['Data'].str[2:5]
new2 = input['Data'].str[15:18]
pd.Series(new1)
pd.Series(new2)
for i in range(len(new1)):
    if i>0:
        if new1[i] == new1[i+1]:
            new1 = new1.drop(index=i)
            new2 = new2.drop(index=i)
BPM = pd.concat([new1,new2], axis=1)
BPM.columns = ['sec','bpm']

BPM.to_csv("D://訓練影片//GO//FCUAI_rppgdata//ex9//{}_bpm.csv".format(num),index=0,header=False)