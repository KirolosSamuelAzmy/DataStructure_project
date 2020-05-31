import pickle
import sys

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

def get_args():
    inputs = []
    for i in range(25):
        inputs.append(sys.argv[i+1])
    query = {}
    query["contains_No"] = inputs[0]
    query["contains_Please"] = inputs[1]
    query["contains_Thank"] = inputs[2]
    query["contains_apologize"] = inputs[3]
    query["contains_bad"] = inputs[4]
    query["contains_clean"] = inputs[5]
    query["contains_comfortable"] = inputs[6]
    query["contains_dirty"] = inputs[7]
    query["contains_enjoyed"] = inputs[8]
    query["contains_friendly"] = inputs[9]
    query["contains_glad"] = inputs[10]
    query["contains_good"] = inputs[11]
    query["contains_great"] = inputs[12]
    query["contains_happy"] = inputs[13]
    query["contains_hot"] = inputs[14]
    query["contains_issues"] = inputs[15]
    query["contains_nice"] = inputs[16]
    query["contains_noise"] = inputs[17]
    query["contains_old"] = inputs[18]
    query["contains_poor"] = inputs[19]
    query["contains_right"] = inputs[20]
    query["contains_small"] = inputs[21]
    query["contains_smell"] = inputs[22]
    query["contains_sorry"] = inputs[23]
    query["contains_wonderful"] = inputs[24]
    query_1 = query
    for i in query:
    	query_1[i] = int(query[i]) 
    return query_1  

def evaluate(query,tree):
	default = 1.0
	result = predict(query,tree,default)
	file = open('verdict','w')
	if result == default:
		file.write("Positive")
	else:
		file.write(result)
	file.close()

query = get_args()
with open('tree','rb') as file:
	tree = pickle.load(file)
evaluate(query,tree)