import pandas as pd

# root = 'D://UBFC-RPPG-dataset//Dataset_1//Cleaned//'
# file_name = '{}-gt.csv'
root = 'D://UBFC-RPPG-dataset//Dataset_2//Cleaned//'
file_name = '{}.csv'

num = 'subject1'
input = pd.read_csv(root+file_name.format(num))

bvp_data = []
x=0


for i in range(len(input)):
    if input['Time'][i] == x*33:
        bvp_data.append(input['Signal'])
        x+=1
    if input['Time'][i] > x*33:
        input['Time'][i]-x*33
        x*33 - input['Time'][i-1]
        
    
# detect_peak = input[input['Peaks'] == 1]

# second =[]
# bpm = []
# for i in range(len(detect_peak)):
#     if i > 0:
#         second.append(detect_peak.iloc[i].Time/1000)
#         bpm.append(60/(detect_peak.iloc[i].Time-detect_peak.iloc[i-1].Time)*1000)
# second = pd.DataFrame(second)
# bpm = pd.DataFrame(bpm)

# # print(bpm)
# bpm_data = pd.concat([second,bpm] ,axis=1)

# bpm_data.to_csv(root+'{}bpm.csv'.format(num),index=0)