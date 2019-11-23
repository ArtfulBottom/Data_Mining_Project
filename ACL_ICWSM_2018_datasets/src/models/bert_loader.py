import pandas as pd
import json
import numpy as np
from numpy import linalg as LA
import time
# import sklearn

def loadBertEmb(bert_features_file_path, numLayers, useAllTokens):
    emb = []
    with open(bert_features_file_path, encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            jsobj  = json.loads(line)
            values = []          
            numTokens = 0
            for tokenId in range(len(jsobj['features'])):
                feature = jsobj['features'][tokenId]                                                     
                layersInfo = feature['layers']
                valuesPerToken = []
                for layerIndex in range(numLayers):
                    hiddenVec  = layersInfo[layerIndex]['values']
                    if len(valuesPerToken)==0 : 
                        valuesPerToken = hiddenVec
                    else:                      
                        valuesPerToken = [valuesPerToken[k] + hiddenVec[k] for k in range(len(valuesPerToken))]
                valuesPerToken =  [valuesPerToken[k]/numLayers for k in range(len(valuesPerToken))]

                if (feature['token'] == '[CLS]') and (useAllTokens == 0):
                    numTokens = 1
                    values    = valuesPerToken                 
                    break
                elif (feature['token'] != '[CLS]') and (useAllTokens==1):   
                    numTokens += 1
                    if len(values)==0:
                        values = valuesPerToken
                    else:
                        values = [values[k] + valuesPerToken[k] for k in range(len(valuesPerToken))]                     
            values = [values[k]/numTokens for k in range(len(values))]           
            emb.append(values)
        
    #convert emb to df
    df = pd.DataFrame(emb, columns = [str(i) for i in range(len(emb[0]))])
    return df

def loadBertEmb2(bert_features_file_path, numLayers, useAllTokens):
    print('baseline')
    emb = []
    with open(bert_features_file_path, encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            jsobj = json.loads(line)
            feature = jsobj['features'][0]            
            values = []
            for i in range(numLayers):
                tmp = feature['layers'][i]['values']
                # print((tmp))
                if len(values)==0 : 
                    values = tmp
                else:                      
                    values = [values[k]+tmp[k] for k in range(len(tmp))]
            values = [values[k]/numLayers for k in range(len(tmp))]
            emb.append(values)
    
    #convert emb to df
    df = pd.DataFrame(emb, columns = [str(i) for i in range(len(emb[0]))])
    return df

if __name__=='__main__':
    # print(sklearn.__version__)
    start = time.time()
    df    = loadBertEmb('../preprocessing/nepal_bert_last_4_layers_all_tokens_base.json', 4, 0)
    df2   = loadBertEmb2('../preprocessing/nepal_bert_last_4_layers_all_tokens_base.json', 4, 0)
    delta = (df-df2).to_numpy()
    end   = time.time()
    print('time:' + str(end - start))
    print('max value in the delta: ' + str(delta.min()) + ", min value: " + str(delta.max()))
    # print(df)