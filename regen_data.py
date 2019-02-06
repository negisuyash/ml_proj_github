import pandas as pd
import os 



def Add_Data(filename):
	df=pd.read_csv('./datasets/%s.csv'%filename,usecols=['addr','d_hub','loc'])
	df=df.dropna()
	'''
	REWRITING WHOLE FILE AGAIN
	old_df=pd.read_csv('./datasets/train_data_full.csv')
	old_df=old_df.append(df)
	old_df.to_csv('./datasets/train_data_full.csv')'''
	df['x_coor']=df['loc'].str.extract(r'.*\[(.*)\,.*')
	df['y_coor']=df['loc'].str.extract(r'.*\,(.*)\].*')

	pd.to_numeric(df['x_coor'])
	pd.to_numeric(df['y_coor'])
	new_breaked_addr=[]

	for i in df['addr']:
		#print(i)
		#new_list=[i]
		#i=remove_values_from_list(i,'gurgaon')
		#i=remove_values_from_list(i,'gurugram')
		#i=remove_values_from_list(i,'')
		#print(i)
		i=i.lower()
		i=i.replace('(',' ')
		i=i.replace(')',' ')
		i=i.replace('-',' ')
		i=i.replace('[',' ')
		i=i.replace(']',' ')
		i=i.replace("'"," ")
		i=i.replace('gurgaon','')
		i=i.replace('gurugram','')
		i=i.replace(',',' ')
		i = ' '.join(filter(None,i.split(' ')))
		new_breaked_addr.append(i)
	#del df['breaked_addr']
	df['breaked_addr']=new_breaked_addr
	#os.system('del datasets\train_data_prep.csv')
	df=df.drop_duplicates()
	if os.path.exists('./datasets/train_data_prep.csv'):
		old_df=pd.read_csv('./datasets/train_data_prep.csv')
		old_df=old_df.append(df)
		old_df.to_csv('./datasets/train_data_prep.csv')
	else:
		print('[ERROR] NO ALREADY PREPARED DATA EXISTS......MAKING NEW DATA')
		Regen_Data()







def Regen_Data():
	df=pd.read_csv('./datasets/export(6).csv',usecols=['loc','d_hub','addr'])
	df1=pd.read_csv('./datasets/train_data_full.csv',usecols=['loc','d_hub','addr'])
	df=df.append(df1)
	del df1
	df1=pd.read_csv('./datasets/hubs.csv')
	df1=df1[df1['temp_off']==0]
	#df=df.dropna()
	df=df.loc[:]
	df1=df1.dropna()

	#geo_of_hub=[]

	#df=df[df['d_hub']!="534e3eef923c4866b75b29ec6dcc0f68"]
	df=df.dropna(axis=0)
	#print(df.loc[150:160])
	'''	DATA PREPARATION FOR HUBS.CSV
	df1['x_coor_hub']=df1['h.location'].str.extract(r'.*\[(.*)\,.*')
	df1['y_coor_hub']=df1['h.location'].str.extract(r'.*\,(.*)\].*')
	df1.to_csv('hubs1.csv')'''

	#FOR TESTING PURPOSE
	#print(df)
	#df.to_csv("cleaned_data1.csv")
	#print(str(len(df))+'\n'+str(len(geo_of_hub)))

	df['x_coor']=df['loc'].str.extract(r'.*\[(.*)\,.*')
	df['y_coor']=df['loc'].str.extract(r'.*\,(.*)\].*')
	df=df.dropna()
	df=df.drop_duplicates()

	pd.to_numeric(df['x_coor'])
	pd.to_numeric(df['y_coor'])
	#df['loc'].str.extract(r'.*\,(.*)\].*'))
	'''list_of_x_coor_hub=[]
	list_of_y_coor_hub=[]
	for i in df['d_hub']:
		for j,k in zip(df1['d.ID'],df1['x_coor_hub']):
			if i==j:
				list_of_x_coor_hub.append()'''
	new_breaked_addr=[]

	for i in df['addr']:
		#print(i)
		#new_list=[i]
		#i=remove_values_from_list(i,'gurgaon')
		#i=remove_values_from_list(i,'gurugram')
		#i=remove_values_from_list(i,'')
		#print(i)
		i=i.lower()
		i=i.replace('(',' ')
		i=i.replace(')',' ')
		i=i.replace('-',' ')
		i=i.replace('[',' ')
		i=i.replace(']',' ')
		i=i.replace("'"," ")
		i=i.replace('gurgaon','')
		i=i.replace('gurugram','')
		i=i.replace(',',' ')
		i = ' '.join(filter(None,i.split(' ')))
		new_breaked_addr.append(i)
	#del df['breaked_addr']
	df['breaked_addr']=new_breaked_addr
	#os.system('del datasets\train_data_prep.csv')
	df=df.drop_duplicates()




	list_of_x_coor=[]
	list_of_y_coor=[]
	list_of_x_coor_hub=[]
	list_of_y_coor_hub=[]
	list_of_hubcode=[]
	list_of_new_hubcode=[]
	list_of_x_coor_hub_regen=[]
	list_of_y_coor_hub_regen=[]
	for i in df['x_coor']:
		list_of_x_coor.append(i)
	for i in df['y_coor']:
		list_of_y_coor.append(i)
	for i in df1['x_coor_hub']:
		list_of_x_coor_hub.append(i)
	for i in df1['y_coor_hub']:
		list_of_y_coor_hub.append(i)
	for i in df1['h.ID']:
		list_of_hubcode.append(i)
	#print(list_of_y_coor_hub)

	for i,j in zip(list_of_x_coor,list_of_y_coor):
		min_diff_x_coor=0XC010
		min_diff_y_coor=0XC010
		new_x_coor=0XC010
		new_y_coor=0XC010
		new_hub_code="THIS IS WRONG CODE"
		for m,n,o in zip(list_of_x_coor_hub,list_of_y_coor_hub,list_of_hubcode):
			
			diff_x_coor=(float(i)-m)**2
			diff_y_coor=(float(j)-n)**2
			if (diff_x_coor+diff_y_coor)<(min_diff_x_coor+min_diff_y_coor):
				min_diff_y_coor=diff_y_coor
				min_diff_x_coor=diff_x_coor
				new_x_coor=m
				new_y_coor=n
				new_hub_code=o
		
		list_of_x_coor_hub_regen.append(new_x_coor)
		list_of_y_coor_hub_regen.append(new_y_coor)
		list_of_new_hubcode.append(new_hub_code)

	#del df['x_coor_hub']
	#del df['y_coor_hub']
	del df['d_hub']
	#df['x_coor_hub']=list_of_x_coor_hub_regen
	#df['y_coor_hub']=list_of_y_coor_hub_regen
	df['d_hub']=list_of_new_hubcode

	print(df)
	#df=pd.read_csv('train_data_full.csv')
	#df=df.dropna()
	
	new_breaked_addr=[]

	for i in df['addr']:
		#print(i)
		#new_list=[i]
		#i=remove_values_from_list(i,'gurgaon')
		#i=remove_values_from_list(i,'gurugram')
		#i=remove_values_from_list(i,'')
		#print(i)
		i=i.lower()
		i=i.replace('(',' ')
		i=i.replace(')',' ')
		i=i.replace('-',' ')
		i=i.replace('[',' ')
		i=i.replace(']',' ')
		i=i.replace("'"," ")
		i=i.replace('gurgaon','')
		i=i.replace('gurugram','')
		i=i.replace(',',' ')
		i = ' '.join(filter(None,i.split(' ')))
		new_breaked_addr.append(i)
	#del df['breaked_addr']
	df['breaked_addr']=new_breaked_addr
	#os.system('del datasets\train_data_prep.csv')
	df=df.drop_duplicates()
	df.to_csv('./datasets/train_data_prep.csv')

def Partial_Regen_Data(old_hub_code,new_hub_code,df):
	df1=pd.read_csv('./datasets/hubs.csv')
	df1=df1[(df1['h.ID']==old_hub_code)|(df1['h.ID']==new_hub_code)]
	
	list_of_x_coor=[]
	list_of_y_coor=[]
	list_of_x_coor_hub=[]
	list_of_y_coor_hub=[]
	list_of_hubcode=[]
	list_of_new_hubcode=[]
	list_of_x_coor_hub_regen=[]
	list_of_y_coor_hub_regen=[]
	for i in df['x_coor']:
		list_of_x_coor.append(i)
	for i in df['y_coor']:
		list_of_y_coor.append(i)
	for i in df1['x_coor_hub']:
		list_of_x_coor_hub.append(i)
	for i in df1['y_coor_hub']:
		list_of_y_coor_hub.append(i)
	for i in df1['h.ID']:
		list_of_hubcode.append(i)
	print(list_of_hubcode)

	for i,j in zip(list_of_x_coor,list_of_y_coor):
		min_diff_x_coor=0XC010
		min_diff_y_coor=0XC010
		new_x_coor=0XC010
		new_y_coor=0XC010
		new_hub_code="THIS IS WRONG CODE"
		for m,n,o in zip(list_of_x_coor_hub,list_of_y_coor_hub,list_of_hubcode):
			
			diff_x_coor=(float(i)-m)**2
			diff_y_coor=(float(j)-n)**2
			if (diff_x_coor+diff_y_coor)<(min_diff_x_coor+min_diff_y_coor):
				min_diff_y_coor=diff_y_coor
				min_diff_x_coor=diff_x_coor
				new_x_coor=m
				new_y_coor=n
				new_hub_code=o
		
		list_of_x_coor_hub_regen.append(new_x_coor)
		list_of_y_coor_hub_regen.append(new_y_coor)
		list_of_new_hubcode.append(new_hub_code)

	#del df['x_coor_hub']
	#del df['y_coor_hub']
	del df['d_hub']
	#df['x_coor_hub']=list_of_x_coor_hub_regen
	#df['y_coor_hub']=list_of_y_coor_hub_regen
	df['d_hub']=list_of_new_hubcode

	#print(df)
	#df=pd.read_csv('train_data_full.csv')
	#df=df.dropna()
	
	new_breaked_addr=[]

	for i in df['addr']:
		#print(i)
		#new_list=[i]
		#i=remove_values_from_list(i,'gurgaon')
		#i=remove_values_from_list(i,'gurugram')
		#i=remove_values_from_list(i,'')
		#print(i)
		i=i.lower()
		i=i.replace('(',' ')
		i=i.replace(')',' ')
		i=i.replace('-',' ')
		i=i.replace('[',' ')
		i=i.replace(']',' ')
		i=i.replace("'"," ")
		i=i.replace('gurgaon','')
		i=i.replace('gurugram','')
		i=i.replace(',',' ')
		i = ' '.join(filter(None,i.split(' ')))
		new_breaked_addr.append(i)
	del df['breaked_addr']
	df['breaked_addr']=new_breaked_addr
	return df


if __name__=='__main__':
	"""df=pd.read_csv('./datasets/train_data_prep.csv',usecols=['x_coor','y_coor','loc','addr','breaked_addr','d_hub'])
				df=df[df['d_hub']=='48b6dba5107b422fb17325c85021dd4f']
				print(len(df))
				print('FUNCTION CALL HERE\n')
				df=Partial_Regen_Data('b1b3de810f1d46619cd868a470ab831c','PALAMWALA',df)
				df.to_csv('test.csv')"""
	Regen_Data()






































#CODED BY SPIDERN3MO