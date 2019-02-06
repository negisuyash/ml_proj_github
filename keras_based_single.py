#!/usr/bin/env python
# coding: utf-8

# In[28]:


import tensorflow as tf
import pandas as pd
import numpy as np
import preprocess_add
from keras.models import Sequential
from keras.layers import Dense
import pickle
import os
import time
from tqdm import tqdm


# In[29]:

def max_finder(l):
	max=l[0]
	for i in l:
		if i>max:
			max=i
	return l.index(max)

#read_data,create new columns
os.system('cls')
os.system('color 2')
df=pd.read_csv('./datasets/train_data_prep.csv',usecols=['breaked_addr','loc','d_hub'])
df=df.dropna()
df1=df.loc[17150:17271]
df=df.loc[:17170]
list_of_hubs=[]
for i in df['d_hub']:
    if i not in list_of_hubs:
        list_of_hubs.append(i)
        
list_of_d=[]
#print(max_finder([2.12,3.45,1.34,0.32]))
        
for i in df['d_hub']:
    list_of_d.append(list_of_hubs.index(i))
    
df['d']=list_of_d   #added indexing of d_hubs
del list_of_d
os.system('echo --------------------------------------------------------')
os.system('echo --------------------------------------------------------')
os.system('echo DATAFRAMES HAVE BEEN LOADED')
os.system('echo --------------------------------------------------------')
#print(df)


# In[31]:


#creating address vectors
input_list=[]
#for i in df['breaked_addr']:
#    print(i)
os.system('echo ------------------------------------------------------------------------')
os.system('echo ------------------------------------------------------------------------')
os.system('echo GENERATING VECTORS FOR LOADED ADDRESSES')
for i,j in zip(df['breaked_addr'],tqdm(range(len(df['breaked_addr'])))):
    input_list.append(preprocess_add.PreProcess_Add(str(i)))
os.system('echo ------------------------------------------------------------------------')
os.system('echo ------------------------------------------------------------------------')
input_x=np.asarray(input_list)

#print(input_list)


# In[32]:


#print(input_list)
df['vector_add']=input_x
#print(df['vector_add'])
df=df.dropna()


# In[33]:


g=globals()
for i in list_of_hubs:
    g['list_%s'%i]=[]
#print(list_of_hubs)
#print(list_b1b3de810f1d46619cd868a470ab831c)
for i in list_of_hubs:
    for j in df['d_hub']:
        if i==j:
            g['list_%s'%i].append(1)
        else:
            g['list_%s'%i].append(0)
#for i in list_of_hubs:
#    print(str(i)+":\n "+str(g['list_%s'%i]))
for i in list_of_hubs:
    df['list_%s'%i]=g['list_%s'%i]
#print(df)


# In[34]:


from tensorflow.python.keras.preprocessing.sequence import pad_sequences
#print(input_list[1])
num_max=[len(tokens) for tokens in input_list]
#print(np.mean(num_max))
#print(np.max(num_max))
pad='pre'
input_list=pad_sequences(input_list,dtype=np.float32,maxlen=int(np.max(num_max)),padding=pad,truncating=pad)
os.system('echo ------------------------------------------------------------------------')
os.system('echo ------------------------------------------------------------------------')
os.system('echo DATA IS PREPARED TO USE NOW')
os.system('echo ------------------------------------------------------------------------')
#print(input_list[1])
#print(input_list.shape)


# In[37]:


#modelling

if os.path.exists('./word_embedding/trained_model.SPIDERN3MO'):
	os.system('echo ------------------------------------------------------------------------')
	os.system('echo ------------------------------------------------------------------------')
	os.system('echo USING PRE-SAVED MODEL')
	os.system('echo ------------------------------------------------------------------------')
	model=pickle.load(open('./word_embedding/trained_model.SPIDERN3MO','rb'))

else:
	os.system('echo ------------------------------------------------------------------------')
	os.system('echo ------------------------------------------------------------------------')
	os.system('echo GENERATING NEW MODEL')
	os.system('echo ------------------------------------------------------------------------')
	model=Sequential()
	#adding layers
	model.add(Dense(800,input_dim=np.max(num_max),activation='relu'))
	model.add(Dense(500,activation='relu'))
	#model.add(Dense(300,activation='relu'))
	model.add(Dense(100,activation='relu'))
	model.add(Dense(len(list_of_hubs),activation='sigmoid'))


	# In[38]:


	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(input_list,df[['list_%s'%i for i in list_of_hubs]].values,epochs=150,batch_size=10)
	pickle.dump(model,open('./word_embedding/trained_model.SPIDERN3MO','wb'))


# In[39]:


scores = model.evaluate(input_list,df[['list_%s'%i for i in list_of_hubs]].values)
print('scores'+str(scores))


# In[42]:


#testing data
#df1=pd.read_csv('./datasets/train_data_prep.csv',usecols=['breaked_addr',''])
#df1=df1.dropna()
#df1=df.loc[:10]
test_list=[]
#df1=df1.dropna()
#for i in df1['breaked_addr']:
#    print(i)
for i in df1['breaked_addr']:
    test_list.append(preprocess_add.PreProcess_Add(str(i)))
    
test_list=pad_sequences(test_list,dtype=np.float32,maxlen=int(np.max(num_max)),padding='pre',truncating='pre')

#print(test_list[1])
prediction=model.predict(test_list)
os.system('echo ------------------------------------------------------------------------')
os.system('echo ------------------------------------------------------------------------')
#print(prediction)
print(list_of_hubs)
correct=0
prediction=prediction.tolist()
for i,j in zip(prediction,df1['d_hub']):
	pred_res=max_finder(i)
	actual_res=list_of_hubs.index(j)
	if pred_res==actual_res:
		correct+=1

print('we got answer:'+str(correct))

add=input('enter address:')
mod_add=[]
mod_add.append(preprocess_add.PreProcess_Add(add))
add=mod_add
add=pad_sequences(add,dtype=np.float32,maxlen=int(np.max(num_max)),padding='pre',truncating='pre')
print(model.predict(add))

# In[ ]:

































































#CODED BY SPIDERN3MO



