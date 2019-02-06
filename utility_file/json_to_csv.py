import json
import urllib.request
import pandas as pd
import os

filename='merchant_1045EB.data'

def response(filename):
	with open(filename) as response:
		return response.read()

add_list=[]

res=response(filename)
#print(res)

#res=json.loads(res)
count=0
#res=res['runsheetList']
for i in res:
	#if ideliveries"]:
	add=""
	for j in i['deliveries']:
		add+=j['consigneeAdd1']+j['consigneeAdd2']+j['consigneeAdd3']
	add_list.append(add)
	print(add)
	print('\n')

#print(add_list)

