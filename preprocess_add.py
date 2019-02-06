import pandas as pd 
import numpy as np 
import get_add_vec
import os


def PreProcess_Add(address):

    if os.path.exists('./datasets/word_embedding.csv'):

    	address=address.split()
    	points=pd.read_csv('./datasets/word_embedding.csv',usecols=['word','x','y'])
    	address_vec=[]
    	for i in address:
    		for j,k,l in zip(points['word'],points['x'],points['y']):
    			if i==j:
    				#address_vec.append([[k],[l]])
    				address_vec.append(k)
    				address_vec.append(l)
    	address_vec=np.asarray(address_vec)
    	return address_vec
    else:
    	print('creating word embeddings...please run program again after completion of execution')
    	get_add_vec.get_add_vec()


if __name__=='__main__':
	add=input('=>')
	print(PreProcess_Add(add))




































































































#CODED BY SPIDERN3MO 