import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score

# cap = cv2.VideoCapture('D://訓練影片//GO//FCUAI_rppgdata//ex6(light+daze)//1-1.mp4')

# fps = cap.get(5)
# frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# print(fps)
# print(frame)
# %%
data = '9-1-exercise'

input = pd.read_csv("C://Users//user//Desktop//{}bpm_data.csv".format(data),index_col=1).transpose()
input_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     


input = input[6:]
input_y = input_y[6:-5].to_numpy()

print(len(input))
print(len(input_y))

data = '9-2-exercise'

input1 = pd.read_csv("C://Users//user//Desktop//{}bpm_data.csv".format(data),index_col=1).transpose()
input1_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     


input1 = input1[6:]
input1_y = input1_y[6:-5].to_numpy()

print(len(input1))
print(len(input1_y))

data = '9-5-exercise'

input2 = pd.read_csv("C://Users//user//Desktop//{}bpm_data.csv".format(data),index_col=1).transpose()
input2_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     

input2 = input2[6:]
input2_y = input2_y[6:-5].to_numpy()


input = pd.concat([input,input1,input2],axis=0)
input_y = np.append(input_y,input1_y)
input_y= np.append(input_y,input2_y)

input = np.array(input).squeeze()
a , b = np.polyfit(input, input_y, 1)

plt.scatter(input,input_y)
plt.xlabel('predict')
plt.ylabel('true')
plt.plot(input, a*input+b,label = 'fitting line', linestyle='--',color='purple')  
plt.plot([0,140],[0,140],label = 'true line',color='orange')
plt.legend(loc = 0)
plt.xlim(50,140)
plt.ylim(50,140)
plt.text(50,142,'RMSE '+str(mean_squared_error(input_y, input)**.5))
plt.text(50,145,'R2 '+str(r2_score(input_y, input)))
plt.show()






# %%
data = '1-1'

input = pd.read_csv("C://Users//user//Desktop//{}Cbpm_data.csv".format(data),index_col=1).transpose()
input_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     


input = input[6:]
input_y = input_y[6:-6].to_numpy()

print(len(input))
print(len(input_y))

data = '2-4'

input1 = pd.read_csv("C://Users//user//Desktop//{}Cbpm_data.csv".format(data),index_col=1).transpose()
input1_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     


input1 = input1[6:]
input1_y = input1_y[6:-5].to_numpy()

print(len(input1))
print(len(input1_y))

data = '3-2'

input2 = pd.read_csv("C://Users//user//Desktop//{}Cbpm_data.csv".format(data),index_col=1).transpose()
input2_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     

input2 = input2[6:]
input2_y = input2_y[6:-5].to_numpy()
print(len(input2))
print(len(input2_y))

data = '3-4'

input3 = pd.read_csv("C://Users//user//Desktop//{}Cbpm_data.csv".format(data),index_col=1).transpose()
input3_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     

input3 = input3[6:]
input3_y = input3_y[6:-5].to_numpy()

print(len(input3))
print(len(input3_y))

data = '6-1-1'

input4 = pd.read_csv("C://Users//user//Desktop//{}Cbpm_data.csv".format(data),index_col=1).transpose()
input4_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     

input4 = input4[6:]
input4_y = input4_y[6:-5].to_numpy()

print(len(input4))
print(len(input4_y))

data = '6-2-2'

input5 = pd.read_csv("C://Users//user//Desktop//{}Cbpm_data.csv".format(data),index_col=1).transpose()
input5_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
     

input5 = input5[6:]
input5_y = input5_y[6:-4].to_numpy()

print(len(input5))
print(len(input5_y))

data = '6-3-2'

input6 = pd.read_csv("C://Users//user//Desktop//{}Cbpm_data.csv".format(data),index_col=1).transpose()
input6_y = pd.read_csv("C://Users//user//Desktop//{}bpm.csv".format(data))
    
input6 = input6[6:]
input6_y = input6_y[6:-5].to_numpy()


input = pd.concat([input,input1,input2,input3,input4,input5,input6],axis=0)
input_y = np.append(input_y,input1_y)
input2_y= np.append(input2_y,input3_y)
input4_y= np.append(input4_y,input5_y)
input4_y= np.append(input4_y,input6_y)
input_y= np.append(input_y,input2_y)
input_y= np.append(input_y,input4_y)

input = np.array(input).squeeze()
a , b = np.polyfit(input, input_y, 1)

plt.scatter(input,input_y)
plt.xlabel('predict')
plt.ylabel('true')
plt.plot(input, a*input+b,label = 'fitting line', linestyle='--',color='purple')  
plt.plot([0,140],[0,140],label = 'true line',color='orange')
plt.legend(loc = 0)
plt.xlim(50,140)
plt.ylim(50,140)
plt.text(50,142,'RMSE '+str(mean_squared_error(input_y, input)**.5))
plt.text(50,145,'R2 '+str(r2_score(input_y, input)))
plt.show()