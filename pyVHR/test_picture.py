from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

data = '9-2-exercise'
resize = 0



# print('predict RMSE：',+mean_squared_error(input_y[1], input[1])**.5)
# print('predict R2：',+r2_score(input_y[1], input[1]))
# print('Kendall tau：'+ str(kendalltau(input_y[1], input[1])))

if resize != True:
    input = pd.read_csv("C://Users//user//Desktop//{}bpm_data.csv".format(data),index_col=1)
    inputC = pd.read_csv("C://Users//user//Desktop//{}框超大bpm_data.csv".format(data),index_col=1)
    input_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
        

    input = np.array(input.transpose()).squeeze()
    inputC = np.array(inputC.transpose()).squeeze()
    input_y = np.array(input_y.transpose()).squeeze()

    input = input[6:]
    inputC = inputC[6:]
    input_y = input_y[7:-4]

    # print('predict RMSE：',+mean_squared_error(input_y[1], input[1])**.5)
    # print('predict R2：',+r2_score(input_y[1], input[1]))
    # print('Kendall tau：'+ str(kendalltau(input_y[1], input[1])))

    print(len(input))
    print(len(inputC))
    print(len(input_y))
    
    plt.plot(input,label='2SR predict')
    plt.plot(inputC,label='2SR no process predict')
    plt.plot(input_y,label='true')

    plt.xlabel("time")
    plt.ylabel("bpm")
    plt.legend(loc = 0)
    plt.xlim(0,len(input))
    plt.ylim(50,140)
    plt.text(0,142,'2SR : RMSE '+str(mean_squared_error(input_y, input)**.5))
    # plt.text(0,145,'CHROM : RMSE '+str(mean_squared_error(input_y, inputC)**.5))
    plt.text(0,148,'2SR : R2 '+str(r2_score(input_y, input)))
    # plt.text(0,151,'CHROM : R2 '+str(r2_score(input_y, inputC)))
    plt.show()
else:
    input = pd.read_csv("C://Users//user//Desktop//{}bpm_data.csv".format(data),index_col=1)
    input72 = pd.read_csv("C://Users//user//Desktop//{}resize72bpm_data.csv".format(data),index_col=1)
    input100 = pd.read_csv("C://Users//user//Desktop//{}resize100bpm_data.csv".format(data),index_col=1)
    input_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
        

    input = np.array(input.transpose()).squeeze()
    input72 = np.array(input72.transpose()).squeeze()
    input100 = np.array(input100.transpose()).squeeze()
    input_y = np.array(input_y.transpose()).squeeze()

    input = input[6:]
    input72 = input72[6:]
    input100 = input100[6:]
    input_y = input_y[6:-5]
    
    plt.plot(input,label='2SR predict')
    plt.plot(input72,label='72*72 predict')
    plt.plot(input100,label='100*100 predict')
    plt.plot(input_y,label='true')

    plt.xlabel("time")
    plt.ylabel("bpm")
    plt.legend(loc = 0)
    plt.xlim(0,len(input))
    plt.ylim(50,140)
    plt.text(0,142,'2SR : RMSE '+str(mean_squared_error(input_y, input)**.5))
    plt.text(0,145,'72*72 : RMSE '+str(mean_squared_error(input_y, input72)**.5))
    plt.text(0,148,'100*100 : RMSE '+str(mean_squared_error(input_y, input100)**.5))
    plt.text(40,142,'2SR : R2 '+str(r2_score(input_y, input)))
    plt.text(40,145,'72*72 : R2 '+str(r2_score(input_y, input72)))
    plt.text(40,148,'100*100 : R2 '+str(r2_score(input_y, input100)))
    plt.show()