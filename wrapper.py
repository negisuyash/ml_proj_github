import keras_based_multiple as KBM 
import pandas as pd 
from tqdm import tqdm


def test_model_acc():
	df=pd.read_csv('./datasets/test_data_prep.csv',usecols=['breaked_addr','d_hub','x_coor','y_coor'])
	df=df.dropna()
	#df=df.loc[20700:]
	correct=0
	#total=1000
	#list_of_hubs=KBM.get_hub_list()
	count=0
	predictions=KBM.predict_data(df,is_block=True)
	for i,j,k,l in zip(predictions[1],df['d_hub'],predictions[0],predictions[2]):
		#d_hub=KBM.predict_data(i)
		#correct_hub=KBM.correct_hub(predictions)
		if i == j:
			correct+=1
		else:
			print(str(count)+" "+str(i)+"  "+str(j)+" "+str(k)+"largest is :"+str(l))
		count+=1
	

	print('acc is :'+str(correct))

def test_model_by_add():
	addr=input("enter string:")
	predictions=KBM.predict_data(addr,is_address=True)
	print(predictions)



if __name__=='__main__':
	print('1.)TO RETRAIN AFTER ADDING NEW HUBS OR REMOVING\n2.)TO GET PREDICTION\n3.)regen data\n4.)SPILT HUB\n5.)DELETE HUB\n6.)ADD NEW HUB\n7.)RUN CURRENT TEST MODULE')
	ch=input('enter choice:')
	if(int(ch)==1):
		KBM.train_data()
	#train_data()
	if(int(ch)==2):
		predictions=KBM.predict_data()
		correct_hub=KBM.correct_hub(predictions)
		list_of_hubs=KBM.get_hub_list()
		print(list_of_hubs[correct_hub])
	if(int(ch)==4):
		KBM.split_hub()
	if(int(ch)==3):
		KBM.regen_data.Regen_Data()
	if(int(ch)==5):
		KBM.delete_hub()
	if(int(ch)==6):
		KBM.add_hub()
	if(int(ch)==7):
		test_model_acc()
	if(int(ch)==8):
		test_model_by_add()