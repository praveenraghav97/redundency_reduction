#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[93]:


'''Random readings of integers for testing purpose... For actual run, please comment this section
df = []
for i in range(1000):
    df.append(np.random.randint(1, 15))

#Comment above for actual run'''


# In[107]:


#The dataset must be preprocessed and scaled as per the given 'error_threshold' value.
#This snippet is taking all the datasets and then making numpy datasets

heart_rate = "HR.csv"
eda = "EDA.csv"
acc = "ACC.csv"
ibi = "IBI.csv"
temp = "TEMP.csv"

#d = np.fromfile(filepath)

df_hr_ = pd.read_csv(heart_rate)
df_eda_ = pd.read_csv(eda)
df_ibi_ = pd.read_csv(ibi)
df_acc_ = pd.read_csv(acc)
df_temp_ = pd.read_csv(temp)

df_list_ = [df_hr_, df_eda_, df_ibi_, df_acc_, df_temp_]



df_hr = df_hr_.to_numpy()
df_eda = df_eda_.to_numpy()
df_ibi= df_ibi_.iloc[:, [1]].to_numpy()
df_acc = df_acc_.iloc[:, [0]].to_numpy() #Accelerometer data only in x-axis
df_acc[:] += 200
df_temp = df_temp_.to_numpy()

df_list = [df_hr, df_eda, df_ibi, df_acc, df_temp]



for df in df_list_:
    print("Size of Dataframe " + str(df.shape))
    print(df.head())
    print(df.describe())


# In[108]:


for df in df_list:
    plt.plot(df)
    plt.show()


# In[109]:


#df_list[2]


# In[110]:


for df in df_list:
    print(len(df))


# In[111]:


k_arr = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80] # Various values for 'k'

#error_thresholds = [e1, e2, e3] # Different threshold values for ECG Signal as per experts.

send = [] # Will store the values which are sent for reference.


# In[112]:


#Random fixed sum generator.

def get_ai_arr(l):
    ai_arr = []
    for t in range(10):      # Generating 10 sets of 'k' random values whose sum equals to '1'        
        a = np.random.rand(l)  
        a = a/np.sum(a, axis=0)
        ai_arr.append(a)
    #print(ai_arr)
    return ai_arr
    
    


# In[113]:


# This method will return the prediction failure value.

def get_pred_fail(ai, df, k, e_th):
    pred_fail = 0
        
    for i in range(k, len(df)):

        yi = df[i]
        yi_pred = 0
        for j in range(0, k):
            yi_pred += ai[j] * df[i - j - 1]

        ei = yi_pred - yi

        yi_pred += ei

        rel_dev = ((abs(yi - yi_pred))/yi) * 100

        if rel_dev > e_th:
            pred_fail += 1
            yi_pred = yi
            send.append(yi)
    return pred_fail
    #print(len(send))

    
    


# In[114]:


# List to store Prediction Faliures for different combinations of 'ai' and 'k'


#This method will find the set of 'ai' which give the optimal value.
#Run this code only for single sensor data..that must be in specific column of dataset as per the Research Paper.
def sampling_frequency_train(df, k, e_th):  
    pred_fail_list = []
    
    ai_arr = get_ai_arr(k)


    for ai in ai_arr:             # Checking for different 'ai' combinations
        send.clear()
        
        for i in range(k):
            send.append(df[i])    # Sending initial K packets        
        
        pred_fail_list.append(get_pred_fail(ai, df, k, e_th))
    
    #print(min(pred_fail_list))
    min_pt = np.argmin(np.array(pred_fail_list))
    p_trans = min_pt/len(df)
    ai_optimal = ai_arr[min_pt]
    #print("Optimal 'ai' Values are: ", ai_optimal)
    return ai_arr, ai_optimal, pred_fail_list    
 


# In[115]:


ai_arr, ai_optimal, pred_fail_list = sampling_frequency_train(df_list[0], 30, 2) # Pass dataframe, value of 'k' and error threshold

plt.plot(pred_fail_list)
plt.show()

pf = get_pred_fail(ai_optimal, df_list[0], 30, 2)
print("Prediction failures For Heart Rate with Error Threshold 2 : ", pf)


p_trans = pf/len(df_list[0])
print("Transmission Probability ", p_trans)


# In[116]:


ai_arr, ai_optimal, pred_fail_list = sampling_frequency_train(df_list[1], 30, .12) # Pass dataframe, value of 'k' and error threshold

plt.plot(pred_fail_list)
plt.show()

pf = get_pred_fail(ai_optimal, df_list[1], 30, .12)
print("Prediction failures For electrodermal activity with Error Threshold .12 : ", pf)


p_trans = pf/len(df_list[1])
print("Transmission Probability ", p_trans)


# In[117]:


ai_arr, ai_optimal, pred_fail_list = sampling_frequency_train(df_list[2], 30, .14) # Pass dataframe, value of 'k' and error threshold

plt.plot(pred_fail_list)
plt.show()

pf = get_pred_fail(ai_optimal, df_list[2], 30, .14)
print("Prediction failures For Inter-Beat Interval with Error Threshold 2 : ", pf)


p_trans = pf/len(df_list[2])
print("Transmission Probability ", p_trans)


# In[118]:


ai_arr, ai_optimal, pred_fail_list = sampling_frequency_train(df_list[3], 30, 3) # Pass dataframe, value of 'k' and error threshold

plt.plot(pred_fail_list)
plt.show()

pf = get_pred_fail(ai_optimal, df_list[3], 30, 3)
print("Prediction failures For Accelerometer with Error Threshold  : ", pf)


p_trans = pf/len(df_list[3])
print("Transmission Probability ", p_trans)


# In[119]:


ai_arr, ai_optimal, pred_fail_list = sampling_frequency_train(df_list[4], 30, .1) # Pass dataframe, value of 'k' and error threshold

plt.plot(pred_fail_list)
plt.show()

pf = get_pred_fail(ai_optimal, df_list[4], 30, .1)
print("Prediction failures For Body Temprature with Error Threshold .1 : ", pf)


p_trans = pf/len(df_list[4])
print("Transmission Probability ", p_trans)


# In[120]:


'''Sampling frequency can be calculated by uncommenting this section of code.'''
#sampling_freq = packet_tras_rate * p_trans                       #----'packet_tras_rate' will depend upon the sensor.


# In[ ]:




