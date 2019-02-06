import googlemaps
import math
import random
import pandas as pd 


gmaps=googlemaps.Client(key="AIzaSyCWLiODnzuTOxu9-75Kv7OAaXIKNMJR2rQ")


list_of_address=[]
list_of_geocode=[]
list_of_hub=[]
c_x=28.457523
c_y=77.026344
c_r=0.3
for i in range(500):
	alpha=2*math.pi * random.random()
	r=c_r*math.sqrt(random.random())
	x=r*math.cos(alpha)+c_x
	y=r*math.sin(alpha)+c_y
	result=gmaps.reverse_geocode((x,y))
	print("x:"+str(x)+"\ty:"+str(y))
	if result is not None:
		list_of_address.append(result[0]['formatted_address'].lower().replace("unnamed road,","").replace(",","").replace("haryana","").replace("india",""))
		list_of_geocode.append("["+str(x)+","+str(y)+"]")
		list_of_hub.append("POWERGRID")

df=pd.DataFrame(list(zip(list_of_address,list_of_geocode,list_of_hub)),columns=['addr','loc','d_hub'])
df.to_csv("test_data.csv")