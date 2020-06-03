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