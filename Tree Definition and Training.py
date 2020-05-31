import numpy as np
import pandas as pd
import csv 
import pickle

def entropy(target_col): 
    
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


def InfoGain(data,split_attribute_name,target_name="rating"):
    
    total_entropy = entropy(data[target_name])
    
    
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def ID3(data,originaldata,features,target_attribute_name="rating",parent_node_class = None):
    
    
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    
    
    elif len(features) ==0:
        return parent_node_class
    
   
    else:
        #Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {best_feature:{}}
        
        
        #Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
        
        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return(tree)    
                
def predict(query,tree):

    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return "Positive" 
  
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
        pred.loc[i,"predicted"] = predict(queries[i],tree) 

    
    
    print('The prediction accuracy is: ',(np.sum(pred["predicted"] == data["rating"])/len(data))*100,'%')
    acc = (np.sum(pred["predicted"] == data["rating"])/len(data))*100
    return pred,acc

def batch_loader(dataset_,batch_size):
    n_batches = dataset_.shape[0]//batch_size
    dataset = dataset_[:n_batches*batch_size]
    for i in range(0,n_batches*batch_size,batch_size):
        yield dataset[0:i+batch_size]

def shuffle(df):
    df = df.sample(frac=1, axis=0).sample(frac=1).reset_index(drop=True)
    return df


#Training on the entire dataset
tree = ID3(dataset,dataset,dataset.columns[:-1])
#______________________________

#Training Loop using mini-batching (initial tree required)
with open("tree", "rb") as f:
    tree = pickle.load(f)
	best_tree = tree
_ , acc_max = test(testdata,tree)
for i,batch in enumerate(batch_loader(dataset,512),1):
        tree = ID3(batch,batch,batch.columns[:-1])
        print("Batch ",i,":")
        pre,acc=test(testdata,tree)
        if acc > acc_max:
            best_tree = tree
            acc_max = acc
#______________________________

#Training Loop using mini-batching and shuffling (initial tree required)
with open("tree", "rb") as f:
    tree = pickle.load(f)
	best_tree = tree
_ , acc_max = test(testdata,tree)
iterations = 4
for ii in range(1,iterations+1):
    print("\nIteration ",ii,":\n")
    shuffled_data = shuffle(dataset)
    for i,batch in enumerate(batch_loader(shuffled_data,512),1):
        tree = ID3(batch,batch,batch.columns[:-1])
        print("Batch ",i,":")
        pre,acc=test(testdata,tree)
        if acc > acc_max:
            best_tree = tree
            acc_max = acc
#______________________________


#Training Loop using mini-batching and shuffling (no intitial tree required)
best_tree = 0
acc_max = 0
iterations = 4
batch_of_max = 1
iteration_of_max = 1
for ii in range(1,iterations+1):
    print("\nIteration ",ii,":\n")
    shuffled_data = shuffle(dataset)
    for i,batch in enumerate(batch_loader(shuffled_data,512),1):
        tree = ID3(batch,batch,batch.columns[:-1])
        print("Batch ",i,":")
        pre,acc=test(testdata,tree)
        if acc > acc_max:
        	batch_of_max = i
        	iteration_of_max = ii
            best_tree = tree
            acc_max = acc
print("\n\nMaximum Accuracy: ",acc_max,"% @Iteration ",iteration_of_max," Batch ",batch_of_max)
#______________________________

#Save tree after training
with open('tree','wb') as f:
    pickle.dump(best_tree,f, pickle.HIGHEST_PROTOCOL)
#______________________________
