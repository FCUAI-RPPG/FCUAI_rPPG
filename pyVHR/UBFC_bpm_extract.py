import pandas as pd

# root = 'D://UBFC-RPPG-dataset//Dataset_1//Cleaned//'
# file_name = '{}-gt.csv'
root = 'D://UBFC-RPPG-dataset//Dataset_2//Cleaned//'
file_name = 'subject{}.csv'

num = '11'
input = pd.read_csv(root+file_name.format(num))

detect_peak = input[input['Peaks'] == 1]

second =[]
bpm = []
for i in range(len(detect_peak)):
    if i > 0:
        second.append(detect_peak.iloc[i].Time/1000)
        bpm.append(60/(detect_peak.iloc[i].Time-detect_peak.iloc[i-1].Time)*1000)
second = pd.DataFrame(second)
bpm = pd.DataFrame(bpm)

# print(bpm)
bpm_data = pd.concat([second,bpm] ,axis=1)

bpm_data.to_csv(root+'{}bpm.csv'.format(num),index=0)