import numpy as np
import pandas as pd
import csv
import pickle
import sys
import time

def predict(query,tree,default = 1):
    
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result

def test(data,tree):

  
    queries = data.iloc[:,:-1].to_dict(orient = 'records')


    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    pred = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        pred.loc[i,"predicted"] = predict(queries[i],tree,1.0) 

    
    pred.predicted.replace(1,"Positive",inplace=True)
    pred.predicted.replace(0,"Negative",inplace=True)

    print('The prediction accuracy is: ',(np.sum(pred["predicted"] == data["rating"])/len(data))*100,'%')
    acc = (np.sum(pred["predicted"] == data["rating"])/len(data))*100
    return pred,acc

test_dataset = sys.argv[1]
dataset = pd.read_csv(test_dataset)
dataset = dataset.drop('reviews.text',axis=1)

with open('tree_new','rb') as file:
    tree = pickle.load(file)

pre,acc=test(dataset,tree)

pre.to_csv(r'aa.txt',  sep=' ', mode='a')

acc = str(acc)
acc = acc.split('.')
acc = acc[0]+'.' +acc[1][0:2]

file = open('test_acc','w')
file.write(str(acc))
file.close()

