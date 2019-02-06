#dependcies
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import os
import pickle
import preprocess_add
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import regen_data
import get_add_vec
import re

#utility functions

def get_hub_list():
	df=pd.read_csv('./datasets/hubs.csv',usecols=['h.ID','temp_off'])
	df=df.dropna()
	list_of_hubs=[]
	for i,j in zip(df['h.ID'],df['temp_off']):
		if i not in list_of_hubs and j==0:
			list_of_hubs.append(i)
	return list_of_hubs

def max_finder(l):
	max=l[0]
	for i in l:
		if i>max:
			max=i
	return l.index(max)

def num_max_gen(l):
	if os.path.exists('./multimodels/num_max/num_max.SPIDERN3MO'):
		return np.max(pickle.load(open('./multimodels/num_max/num_max.SPIDERN3MO','rb')))
	else:
		return 0

def correct_hub(predictions):
	largest=float(predictions[0])
	#largest_id=0
	for i in predictions:
		if largest < float(i):
			largest=float(i)
	return predictions.index(largest)



def train_data(new_hub_name=None,add_hub=False):

	df=pd.read_csv('./datasets/train_data_prep.csv',usecols=['breaked_addr','d_hub','loc'])
	df=df.dropna()
	df=df.loc[:20700]
	#if add_hub is False:
	#	os.system('del multimodels\*.SPIDERN3MO')
	
	
	

	list_of_hubs=get_hub_list()
	'''input_list=[]
	print('GENERATING VECTORS:' )
	for k,j in zip(training_data['breaked_addr'],tqdm(range(len(training_data['breaked_addr'])))):
		input_list.append(preprocess_add.PreProcess_Add(str(k)))'''

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

	training_data=df.loc[:]
	testing_data=df.loc[17150:17271]
	#TRAINING HERE////////
	if add_hub is False:
		for i in range(len(list_of_hubs)):
			#if i == (len(list_of_hubs)-1):
			#training_data=training_data[training_data['d_hub']==list_of_hubs[i-1]].head(50)
			temp_training_data=training_data[(training_data['d_hub']==list_of_hubs[i])] #| (((training_data['d_hub']==list_of_hubs[j]).head(500)) for j in range(len(list_of_hubs)) )]
			#else:
			#training_data=training_data[training_data['d_hub']==list_of_hubs[i+1]].head(50)
			#	temp_training_data=training_data[(training_data['d_hub']==list_of_hubs[i])] #|  (((training_data['d_hub']==list_of_hubs[j]).head(500))for j in range(len(list_of_hubs)) )]
			length_temp_training_data=len(temp_training_data)
			for j in range(len(list_of_hubs)):
				if j!=i:
					temp_training_data=temp_training_data.append(training_data[training_data['d_hub']==list_of_hubs[j]].head(int(1.8*(length_temp_training_data)/(len(list_of_hubs)))))
					#print()

			#print(len(temp_training_data))
			#temp_training_data.to_csv('test.csv')
			#h=input('enter any thing:')
			temp_training_data.to_csv('%s_data.csv'%list_of_hubs[i])
			if len(temp_training_data)<8500:
				epochs=7
			elif len(temp_training_data)>8500:
				epochs=int(9+(len(temp_training_data)/6000)*2)

			input_list=[]
			#TESTING
			
			new_breaked_addr=[]
			for j in temp_training_data['breaked_addr']:
				k=re.findall(r"[^\W\d_]+|\d+", j)
				temp_add=""
				for x in k:
					temp_add+=x+" "
				k=temp_add.strip()
				new_breaked_addr.append(k)
				#print(k)
			del temp_training_data['breaked_addr']
			temp_training_data['breaked_addr']=new_breaked_addr
			del new_breaked_addr

			#TILL HERE
			
			print('GENERATING VECTORS for %s:'%list_of_hubs[i] )
			count=0
			for k in temp_training_data['breaked_addr']:
				print(count)
				count+=1
				input_list.append(preprocess_add.PreProcess_Add(str(k)))

		

		
			pad='pre'

			#output_list=[]
			#for j in range(len(temp_training_data)):
				#output_list.append(1)
			is_num_max=num_max_gen(input_list)
			if is_num_max==0:
				is_num_max=[len(tokens) for tokens in input_list]

			input_list=pad_sequences(input_list,dtype=np.float32,maxlen=int(np.max(is_num_max)),padding=pad,truncating=pad)
			model=Sequential()
			#adding layers
			model.add(Dense(1000,input_dim=np.max(is_num_max),activation='relu'))
			model.add(Dense(800,activation='relu'))
			model.add(Dense(500,activation='relu'))
			model.add(Dense(300,activation='relu'))
			model.add(Dense(100,activation='relu'))
			#model.add(Dense(50,activation='relu'))
			model.add(Dense(1,activation='sigmoid'))


			# In[38]:
		
			#print('%s'%i)
			output_list=[]
			for j in temp_training_data['list_%s'%list_of_hubs[i]]:
				output_list.append(j)
			print(len(output_list))
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			model.fit(input_list,output_list,epochs=epochs,batch_size=33)
			pickle.dump(model,open('./multimodels/model_%s.SPIDERN3MO' %list_of_hubs[i],'wb'))
			if os.path.exists('./multimodels/num_max/num_max.SPIDERN3MO') and is_num_max==0:
				if np.max(num_max)>np.max(pickle.load(open('./multimodels/num_max/num_max.SPIDERN3MO','rb'))):
					pickle.dump(num_max,open('./multimodels/num_max/num_max.SPIDERN3MO','wb'))
			elif is_num_max==0:
				pickle.dump(num_max,open('./multimodels/num_max/num_max.SPIDERN3MO','wb'))
	if add_hub is True:
		temp_training_data=training_data[(training_data['d_hub']==new_hub_name)] #| (((training_data['d_hub']==list_of_hubs[j]).head(500)) for j in range(len(list_of_hubs)) )]
		#else:
		#training_data=training_data[training_data['d_hub']==list_of_hubs[i+1]].head(50)
		#	temp_training_data=training_data[(training_data['d_hub']==list_of_hubs[i])] #|  (((training_data['d_hub']==list_of_hubs[j]).head(500))for j in range(len(list_of_hubs)) )]
		length_temp_training_data=len(temp_training_data)
		for j in range(len(list_of_hubs)):
			if j!=new_hub_name:
				temp_training_data=temp_training_data.append(training_data[training_data['d_hub']==list_of_hubs[j]].head(int((length_temp_training_data)/(len(list_of_hubs)))))
				#print()

		#print(len(temp_training_data))
		#temp_training_data.to_csv('test.csv')
		#h=input('enter any thing:')

		input_list=[]
		print('GENERATING VECTORS for %s:'%new_hub_name )
		for k,j in zip(temp_training_data['breaked_addr'],tqdm(range(len(temp_training_data['breaked_addr'])))):
			input_list.append(preprocess_add.PreProcess_Add(str(k)))
		

		
		pad='pre'

		#output_list=[]
		#for j in range(len(temp_training_data)):
		#output_list.append(1)
		is_num_max=num_max_gen(input_list)
		if is_num_max==0:
			is_num_max=[len(tokens) for tokens in input_list]

		input_list=pad_sequences(input_list,dtype=np.float32,maxlen=int(np.max(is_num_max)),padding=pad,truncating=pad)
		model=Sequential()
		#adding layers
		model.add(Dense(800,input_dim=np.max(is_num_max),activation='relu'))
		model.add(Dense(500,activation='relu'))
		model.add(Dense(300,activation='relu'))
		model.add(Dense(100,activation='relu'))
		model.add(Dense(1,activation='sigmoid'))


		# In[38]:
					#print('%s'%i)
		output_list=[]
		for j in temp_training_data['list_%s'%new_hub_name]:
			output_list.append(j)
		print(len(output_list))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(input_list,output_list,epochs=10,batch_size=15)
		pickle.dump(model,open('./multimodels/model_%s.SPIDERN3MO' %new_hub_name,'wb'))
		if os.path.exists('./multimodels/num_max/num_max.SPIDERN3MO') and is_num_max==0:
			if np.max(num_max)>np.max(pickle.load(open('./multimodels/num_max/num_max.SPIDERN3MO','rb'))):
				pickle.dump(num_max,open('./multimodels/num_max/num_max.SPIDERN3MO','wb'))
		elif is_num_max==0:
			pickle.dump(num_max,open('./multimodels/num_max/num_max.SPIDERN3MO','wb'))



def predict_data(address=None,is_block=False,is_address=False):
	list_of_hubs=get_hub_list()
	if is_address is True:
		address=input('ENTER THE ADDRESS STRING:')
	if is_block is False and is_address is True:
		predictions=[]
		mod_add=[]
		#TESTING
		address=re.findall(r"[^\W\d_]+|\d+", address)
		temp_add=""
		for i in address:
			temp_add+=i+" "
		address=temp_add.strip()
		print(address)

		#TILL HERE

		mod_add.append(preprocess_add.PreProcess_Add(address))
		num_max=pickle.load(open('./multimodels/num_max/num_max.SPIDERN3MO','rb'))
		mod_add=pad_sequences(mod_add,dtype=np.float32,maxlen=int(np.max(num_max)),padding='pre',truncating='pre')

		g=globals()
		for i in list_of_hubs:
			g['%s'%i]=pickle.load(open('./multimodels/model_%s.SPIDERN3MO'%i,'rb'))
		#print(list_of_hubs)
		#print(list_b1b3de810f1d46619cd868a470ab831c)
	

		for i in range(len(list_of_hubs)):
		
			#model=pickle.load(open('./multimodels/model_%s.SPIDERN3MO'%list_of_hubs[i],'rb'))
			#print('prediction for %s:'%list_of_hubs[i])
		
			#print(model.predict(mod_add))
			model=g['%s'%list_of_hubs[i]]
			predictions.append(model.predict(mod_add))
		ind=predictions.index(max(predictions))
		return list_of_hubs[ind]
	elif is_block is True and is_address is False:
		predictions=[]
		predictions_1=[]
		predictions_2=[]
		mod_add=[]
		#TESTING
		new_breaked_addr=[]
		for j in address['breaked_addr']:
			k=re.findall(r"[^\W\d_]+|\d+", j)
			temp_add=""
			for i in k:
				temp_add+=i+" "
			k=temp_add.strip()
			new_breaked_addr.append(k)
			print(k)
		del address['breaked_addr']
		address['breaked_addr']=new_breaked_addr
		del new_breaked_addr

		#TILL HERE
		for i,j in zip(address['breaked_addr'],tqdm(range(len(address)))):
			mod_add.append(preprocess_add.PreProcess_Add(i))
		num_max=pickle.load(open('./multimodels/num_max/num_max.SPIDERN3MO','rb'))
		mod_add=pad_sequences(mod_add,dtype=np.float32,maxlen=int(np.max(num_max)),padding='pre',truncating='pre')

		
		g=globals()
		for i in list_of_hubs:
			g['%s'%i]=pickle.load(open('./multimodels/model_%s.SPIDERN3MO'%i,'rb'))
		#print(list_of_hubs)
		#print(list_b1b3de810f1d46619cd868a470ab831c)
		predictions_3=[]
		list_of_x_coors=[]
		list_of_y_coors=[]
		list_of_wrong_hubs=[]
		count=0
		for i,x,y,z in zip(mod_add,address['x_coor'],address['y_coor'],address['d_hub']):
		
			#model=pickle.load(open('./multimodels/model_%s.SPIDERN3MO'%list_of_hubs[i],'rb'))
			#print('prediction for %s:'%list_of_hubs[i])
		
			#print(model.predict(mod_add))
			temp_predictions=[]
			#print("shape:"+str(np.shape(i)))
			#print(count)
			count+=1
			for j in range(len(list_of_hubs)):
				
				#model=g['%s'%list_of_hubs[j]]
				#print(type(i))
				model=g['%s'%list_of_hubs[j]]
				#print("shape:"+str(np.shape(i)))
				temp_predictions.append(float(model.predict(np.array([i]))))
			predictions_1.append(temp_predictions)
			ind=temp_predictions[max_finder(temp_predictions)]
			predictions_3.append(max_finder(temp_predictions))
			predictions_2.append(list_of_hubs[temp_predictions.index(ind)])
			if list_of_hubs[temp_predictions.index(ind)] != z:
				list_of_x_coors.append(x)
				list_of_y_coors.append(y)
				list_of_wrong_hubs.append(list_of_hubs[temp_predictions.index(ind)])

		points=pd.DataFrame(columns=['x_coor','y_coor','d_hub'])
		print(str(len(list_of_x_coors))+" "+str(len(list_of_y_coors))+" "+str(len(list_of_wrong_hubs)))
		points['x_coor']=list_of_x_coors
		points['y_coor']=list_of_y_coors
		points['d_hub']=list_of_wrong_hubs
		points.to_csv('./datasets/wrong_output.csv')


		predictions=[predictions_1,predictions_2,predictions_3]
		return predictions


def delete_hub():
	delete_hub_code=input("HUB CODE YOU WANT TO DELETE:")
	merge_hub_code=input("HUB CODE YOU WANT TO MERGE WITH:")
	df=pd.read_csv('./datasets/train_data_prep.csv')
	#print('still here')
	df1=df[df['d_hub']==delete_hub_code]
	df=df[df['d_hub']!=delete_hub_code]
	temp_list=[]
	for i in range(len(df1)):
		temp_list.append(merge_hub_code)
	del df1['d_hub']
	df1['d_hub']=temp_list
	df=df.append(df1)
	partial_train_model(df,merge_hub_code,delete_hub_code,delete=True)
	del df
	del temp_list
	df=pd.read_csv('./datasets/hubs.csv',usecols=['h.ID','temp_off'])
	temp_list=[]
	for i,j in zip(df['h.ID'],df['temp_off']):
		if i==delete_hub_code:
			temp_list.append(1)
		else:
			temp_list.append(j)
	del df['temp_off']
	df['temp_off']=temp_list
	df.to_csv('./datasets/hubs.csv')





def test_acc_module():
	list_of_hubs=get_hub_list()
	df=pd.read_csv('./datasets/train_data_prep.csv')
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
	input_list=[]
	testing_data=df['breaked_addr'].loc[:600]
	for j,k in zip(testing_data,tqdm(range(len(testing_data)))):
		input_list.append(preprocess_add.PreProcess_Add(str(j)))
	num_max=pickle.load(open('./multimodels/num_max/num_max.SPIDERN3MO','rb'))
	input_list=pad_sequences(input_list,dtype=np.float32,maxlen=int(np.max(num_max)),padding='pre',truncating='pre')
	for i in list_of_hubs:
		model=pickle.load(open('./multimodels/model_%s.SPIDERN3MO'%i,'rb'))
		
		output_list=[]
		
		#print(testing_data)
		
		testing_output=df['list_%s'%i].loc[:600]
		
		
		for j in testing_output:
			output_list.append(j)
		scores = model.evaluate(input_list,output_list)
		print('FOR %s'%i)
		print(scores)
		#print("Baseline Error: %.2f%%" % (100-scores[1]*100))
 



def split_hub():
	old_hub_code=input('ENTER HUB CODE HERE:')
	new_hub_code=input('ENTER NEW HUB CODE HERE:')
	df=pd.read_csv('./datasets/train_data_prep.csv')
	df=df.dropna()
	df1=df[df['d_hub']==old_hub_code]
	df1=regen_data.Partial_Regen_Data(old_hub_code,new_hub_code,df1)
	df=df[df['d_hub']!=old_hub_code]
	df=df.append(df1)
	#df.to_csv('test.csv')
	partial_train_model(df,old_hub_code,new_hub_code)

def partial_train_model(df,old_hub_code,new_hub_code,delete=False):
	if delete==True:
		os.system('del multimodels\model_%s.SPIDERN3MO'%new_hub_code)
	df=df[['d_hub','addr','breaked_addr','loc','x_coor','y_coor']]
	df.to_csv('./datasets/train_data_prep.csv')
	list_of_hubs=get_hub_list()
	print(list_of_hubs)
	'''input_list=[]
	print('GENERATING VECTORS:' )
	for k,j in zip(training_data['breaked_addr'],tqdm(range(len(training_data['breaked_addr'])))):
		input_list.append(preprocess_add.PreProcess_Add(str(k)))'''

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
	#df.to_csv('test.csv')
	df=df.drop_duplicates()

	training_data=df
	counter=0
	#i=0
	flag=False
	i=0
	for i in range(0,len(list_of_hubs)):
	#while (i >=0) and (i<len(list_of_hubs)):
		if (list_of_hubs[i]==new_hub_code and delete==False) or (list_of_hubs[i]==old_hub_code and delete==True):
			counter=1
			#print("in loop")

			temp_training_data=training_data[(training_data['d_hub']==list_of_hubs[i])] 
			length_temp_training_data=len(temp_training_data)
			for j in range(len(list_of_hubs)):
				if j!=i and list_of_hubs[j]!=old_hub_code:
					temp_training_data=temp_training_data.append(training_data[training_data['d_hub']==list_of_hubs[j]].head(int(2*length_temp_training_data/(3*len(list_of_hubs)))))
			#temp_training_data.to_csv('test1.csv')
			temp_training_data=temp_training_data.append(training_data[training_data['d_hub']==old_hub_code])
			temp_training_data.to_csv('temp_training_data.csv')
			input_list=[]
			print('[INFO] GENERATING VECTORS for %s:'%list_of_hubs[i] )
			for k,j in zip(temp_training_data['breaked_addr'],tqdm(range(len(temp_training_data['breaked_addr'])))):
				input_list.append(preprocess_add.PreProcess_Add(str(k)))
			pad='pre'
			is_num_max=num_max_gen(input_list)
			if is_num_max==0:
				is_num_max=[len(tokens) for tokens in input_list]
			input_list=pad_sequences(input_list,dtype=np.float32,maxlen=int(np.max(is_num_max)),padding=pad,truncating=pad)
			model=Sequential()
			#adding layers
			model.add(Dense(800,input_dim=np.max(is_num_max),activation='relu'))
			model.add(Dense(500,activation='relu'))
			#model.add(Dense(300,activation='relu'))
			model.add(Dense(100,activation='relu'))
			model.add(Dense(1,activation='sigmoid'))
			# In[38]:
			print('%s'%i)
			output_list=[]
			for j in temp_training_data['list_%s'%list_of_hubs[i]]:
				output_list.append(j)
			#print(len(output_list))
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			model.fit(input_list,output_list,epochs=20,batch_size=13)
			pickle.dump(model,open('./multimodels/model_%s.SPIDERN3MO' %list_of_hubs[i],'wb'))
			if list_of_hubs[i]==list_of_hubs[len(list_of_hubs)-1]:
				print('hey there')
				flag=True
			if flag==True:
				print('test 2')
				i-=1
			elif flag==False:
				i+=1


			if os.path.exists('./multimodels/num_max/num_max.SPIDERN3MO') and is_num_max==0:
				if np.max(num_max)>np.max(pickle.load(open('./multimodels/num_max/num_max.SPIDERN3MO','rb'))):
					pickle.dump(num_max,open('./multimodels/num_max/num_max.SPIDERN3MO','wb'))
			elif is_num_max==0:
				pickle.dump(num_max,open('./multimodels/num_max/num_max.SPIDERN3MO','wb'))

			if flag==True and i==0:
				break



	for i in list_of_hubs:
		if (i==old_hub_code and counter==1 and delete==False):
			os.system('del multimodels\model_%s.SPIDERN3MO'%i)
			temp_training_data=training_data[(training_data['d_hub']==i)] 
			length_temp_training_data=len(temp_training_data)
			for j in range(len(list_of_hubs)):
				if list_of_hubs[j]!=i and list_of_hubs[j]!=new_hub_code:
					temp_training_data=temp_training_data.append(training_data[training_data['d_hub']==list_of_hubs[j]].head(int(2*length_temp_training_data/(3*len(list_of_hubs)))))
			#temp_training_data.to_csv('test1.csv')
			temp_training_data=temp_training_data.append(training_data[training_data['d_hub']==new_hub_code])
			temp_training_data.to_csv('temp_training_data1.csv')
			input_list=[]
			print('[INFO] GENERATING VECTORS for %s:'%i )
			for k,j in zip(temp_training_data['breaked_addr'],tqdm(range(len(temp_training_data['breaked_addr'])))):
				input_list.append(preprocess_add.PreProcess_Add(str(k)))
			pad='pre'
			is_num_max=num_max_gen(input_list)
			if is_num_max==0:
				is_num_max=[len(tokens) for tokens in input_list]
			input_list=pad_sequences(input_list,dtype=np.float32,maxlen=int(np.max(is_num_max)),padding=pad,truncating=pad)
			model=Sequential()
			#adding layers
			model.add(Dense(800,input_dim=np.max(is_num_max),activation='relu'))
			model.add(Dense(500,activation='relu'))
			#model.add(Dense(300,activation='relu'))
			model.add(Dense(100,activation='relu'))
			model.add(Dense(1,activation='sigmoid'))
			# In[38]:
			print('%s'%i)
			output_list=[]
			for j in temp_training_data['list_%s'%i]:
				output_list.append(j)
			#print(len(output_list))
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			model.fit(input_list,output_list,epochs=30,batch_size=13)
			pickle.dump(model,open('./multimodels/model_%s.SPIDERN3MO' %i,'wb'))
			#if list_of_hubs[i]==list_of_hubs[len(list_of_hubs)-1]:
			#	print('hey there')
			#	flag=True
			#if flag==True:
			#	print('test 2')
			#	i-=1
			#elif flag==False:
			#	i+=1


			if os.path.exists('./multimodels/num_max/num_max.SPIDERN3MO') and is_num_max==0:
				if np.max(num_max)>np.max(pickle.load(open('./multimodels/num_max/num_max.SPIDERN3MO','rb'))):
					pickle.dump(num_max,open('./multimodels/num_max/num_max.SPIDERN3MO','wb'))
			elif is_num_max==0:
				pickle.dump(num_max,open('./multimodels/num_max/num_max.SPIDERN3MO','wb'))

			if flag==True and i==0:
				break

		#else:




#def delete_h
#
#	hub_code=input('ENTER HUB YOU WANT TO DELETE:')



def add_hub():
	df=pd.read_csv('./datasets/new_hub.csv',usecols=['addr','d_hub','loc'])
	get_add_vec.get_add_vec("new_hub")
	regen_data.Add_Data('new_hub')
	new_hub_name=input('ENTER NEW HUB NAME:')
	train_data(new_hub_name=new_hub_name,add_hub=True)










if __name__=='__main__':
	print('1.)TO RETRAIN AFTER ADDING NEW HUBS OR REMOVING\n2.)TO GET PREDICTION\n3.)regen data\n4.)SPILT HUB\n5.)DELETE HUB\n6.)ADD NEW HUB\n7.)RUN CURRENT TEST MODULE')
	ch=input('enter choice:')
	if(int(ch)==1):
		train_data()
	#train_data()
	if(int(ch)==2):
		predict_data()
	if(int(ch)==4):
		split_hub()
	if(int(ch)==3):
		add_hub()
	if(int(ch)==5):
		delete_hub()
	if(int(ch)==6):
		add_hub()
	if(int(ch)==7):
		test_acc_module()
	































































#CODED BY SPIDERN3MO
