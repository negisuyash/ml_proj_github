import gmplot
import pandas as pd 
import numpy as np
import keras_based_multiple as KBM

def MapWrap():

	list_of_hubs=KBM.get_hub_list()

	df=pd.read_csv('./datasets/wrong_output.csv',usecols=['x_coor','y_coor','d_hub'])
	df1=pd.read_csv('./datasets/hubs.csv',usecols=['h.ID','temp_off'])
	#df['x_coor']=df['loc'].str.extract(r'.*\[(.*)\,.*')
	#df['y_coor']=df['loc'].str.extract(r'.*\,(.*)\].*')
	#del df['loc']
	#df=df.loc[:16000,['d_hub','x_coor','y_coor']]
	g=globals()

	#for i in list_of_hubs:
	#	g['df_%s'%i]=df[df['d_hub']==i]
	#if df['d_hub'].str.contains("b1b3de810f1d46619cd868a470ab831c"):

	for i in list_of_hubs:
		temp_df=df[df['d_hub']==i]
		temp_df_lat=temp_df['x_coor']
		temp_df_lon=temp_df['y_coor']
		g['latitude_list_%s'%i]=np.array(temp_df_lat.values,dtype='float32')
		g['longitude_list_%s'%i]=np.array(temp_df_lon.values,dtype='float32')


	'''latitude_list_0=np.array(df0['x_coor'].as_matrix(),dtype='float32') 
	longitude_list_0 =np.array(df0['y_coor'].as_matrix(),dtype='float32') 

	#if df['d_hub'].str.contains("48b6dba5107b422fb17325c85021dd4f"):
	latitude_list_1=np.array(df1['x_coor'].as_matrix(),dtype='float32') 
	longitude_list_1 =np.array(df1['y_coor'].as_matrix(),dtype='float32')

	latitude_list_2=np.array(df2['x_coor'].as_matrix(),dtype='float32') 
	longitude_list_2 =np.array(df2['y_coor'].as_matrix(),dtype='float32') 

	latitude_list_3=np.array(df3['x_coor'].as_matrix(),dtype='float32') 
	longitude_list_3 =np.array(df3['y_coor'].as_matrix(),dtype='float32') 

	latitude_list_4=np.array(df4['x_coor'].as_matrix(),dtype='float32') 
	longitude_list_4 =np.array(df4['y_coor'].as_matrix(),dtype='float32')  

	latitude_list_5=np.array(df5['x_coor'].as_matrix(),dtype='float32') 
	longitude_list_5 =np.array(df5['y_coor'].as_matrix(),dtype='float32')''' 

	#print(latitude_list_1)
	gmap3 = gmplot.GoogleMapPlotter(28.4595, 77.0266, 12)
	'''gmap3.scatter(latitude_list_0,longitude_list_0,'#FF0000',size=40,marker=False)
	gmap3.scatter(latitude_list_1,longitude_list_1,'#00FF00',size=40,marker=False)
	gmap3.scatter(latitude_list_2,longitude_list_2,'#FFFF00',size=40,marker=False)
	gmap3.scatter(latitude_list_3,longitude_list_3,'#0000FF',size=40,marker=False)
	gmap3.scatter(latitude_list_4,longitude_list_4,'#FF00FF',size=40,marker=False)
	gmap3.scatter(latitude_list_5,longitude_list_5,'#00FFFF',size=40,marker=False)'''
	
	for i in list_of_hubs:
		if i=="b1b3de810f1d46619cd868a470ab831c":
			color='#FF0000'
		#elif i=="PALAMWALA":
		#	color='#00FF00'
		elif i=="48b6dba5107b422fb17325c85021dd4f":
			color='#FFFF00'
		elif i=="be7261b61da149aa805e10f395c46ed3":
			color='#0000FF'
		#elif i=="S49":
		#	color='#FF00FF'
		#elif i=="DLFCYBERHUB":
		#	color='$00FFFF'
		gmap3.scatter(g['latitude_list_%s'%i],g['longitude_list_%s'%i],color,size=40,marker=False)
	gmap3.scatter([28.48581],[77.0192451],'#FF0000',size=240,marker=False)
	gmap3.scatter([28.4861289],[77.0620486],'#00FF00',size=240,marker=False)
	gmap3.scatter([28.5052549],[77.0381478],'#FFFF00',size=240,marker=False)
	gmap3.scatter([28.414905],[77.0348477],'#0000FF',size=240,marker=False)
	gmap3.scatter([28.4195442],[77.0493578],'#FF00FF',size=240,marker=False)
	gmap3.scatter([28.4901695],[77.0953785],'#00FFFF',size=240,marker=False)
	#gmap3.scatter(latitude_list_6,longitude_list_6,'#FF00FF',size=40,marker=False)
	#gmap3.plot(latitude_list_0,longitude_list_0,'orange',edge_width=2.5)
	gmap3.draw('F:\\qfrog\\ml_beta\\NN_approach\\map.html') 
	#F:\qfrog\ml_beta\geocode\templates
if __name__=='__main__':
	MapWrap()