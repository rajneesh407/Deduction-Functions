import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import threading
import time
from sklearn.neighbors import NearestNeighbors
class KNNFeature(object):
    def __init__(self,train,test):
        self.train=train
        self.test=test


    def normalizeAmount(self,i,clusterset,amount_column,amount,check):
        mean_amount=clusterset[amount_column].mean()

        if(check==1):
            self.train.loc[self.train.index == i, 'normalize_amount'] =(amount/mean_amount)
        if(check==2):
            self.test.loc[self.test.index == i, 'normalize_amount'] = (amount / mean_amount)

    def knnHistory(self,i,clusterset,output,pk_feature,check):
        total=clusterset[pk_feature].count()
        valid=clusterset[output].sum()

        if (check == 1):
            self.train.loc[self.train.index == i, 'knn_history']  = (total-valid)/total
        if (check == 2):
            self.test.loc[self.test.index == i, 'knn_history']=(total-valid)/total


    def knnB_Value(self,i,clusterset,amount_column,output,amount,check):
        average_dispute_amount=clusterset[amount_column].mean()
        mean_invalid_amount=clusterset[clusterset[output]==0][amount_column].mean()
        if(mean_invalid_amount/average_dispute_amount>1):

            if (check == 1):
                self.train.loc[self.train.index == i, 'knn_b_value'] =amount/mean_invalid_amount
            if (check == 2):
                self.test.loc[self.test.index == i, 'knn_b_value'] = amount/mean_invalid_amount

        else:


            if (check == 1):
                self.train.loc[self.train.index == i, 'knn_b_value'] =1/(amount/mean_invalid_amount)
            if (check == 2):
                self.test.loc[self.test.index == i, 'knn_b_value'] = 1/(amount/mean_invalid_amount)


    def findCluster(self,neighbors,idx,d1,feature_set):
        x = neighbors.kneighbors(d1[feature_set], return_distance=False)
        x = x.tolist()[0]
        if (idx in x):
            a = x.index(idx)
            del x[a]
        return(self.train.iloc[x])

    def kNNTrain(self,idx,neighbors,amount_column,d1,feature_set,output,pk_feature):
        i=idx
        clusterset = KNNFeature.findCluster(self, neighbors, i, d1, feature_set)
        KNNFeature.normalizeAmount(self,i,clusterset,amount_column,d1[amount_column],1)
        KNNFeature.knnHistory(self,i,clusterset,output,pk_feature,1)
        KNNFeature.knnB_Value(self,i,clusterset,amount_column,output,d1[amount_column],1)
        print(self.train.loc[self.train.index == i, ['knn_b_value', 'normalize_amount', 'knn_history']])

    def kNNTest(self,idx,neighbors,amount_column,d1,feature_set,output,pk_feature):
        i = idx
        clusterset = KNNFeature.findCluster(self, neighbors, i, d1, feature_set)
        KNNFeature.normalizeAmount(self, i, clusterset,amount_column,d1[amount_column],2)
        KNNFeature.knnHistory(self, i, clusterset, output,pk_feature,2)
        KNNFeature.knnB_Value(self, i, clusterset, amount_column,output, d1[amount_column],2)
        print(self.test.loc[self.test.index == i, ['knn_b_value','normalize_amount','knn_history']])

    def createKNNFeature(self,feature_set,amount_column,output,pk_feature,k):
        self.train['normalize_amount']=np.nan
        self.train['knn_b_value']=np.nan
        self.train['knn_history']=np.nan
        self.test['normalize_amount'] = np.nan
        self.test['knn_b_value'] = np.nan
        self.test['knn_history'] = np.nan
        index_list = self.train.index.tolist()
        index_list1= self.test.index.tolist()
        count=0
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.train[feature_set])
        while (len(index_list1)!= 0):
            if (len(index_list1)>= 6):
                temp_list = index_list1[:6]
            else:
                temp_list = index_list1
            thread_list = []
            count=count+len(temp_list)
            print(count)
            for i in range(len(temp_list)):
                d1 = self.test[self.test.index == temp_list[i]]
                t = threading.Thread(target=KNNFeature.kNNTest, name='thread{}'.format(i),
                                     args=(self, temp_list[i], neighbors,
                                           amount_column, d1, feature_set, output,
                                           pk_feature))
                thread_list.append(t)
                t.start()
                time.sleep(0.05)
            for t in thread_list:
                t.join()
            index_list1 = [x for x in index_list1 if x not in temp_list]

        print("test complete")
        count=0
        while (len(index_list)!=0):
            if(len(index_list)>=6):
                temp_list=index_list[:6]
            else:
                temp_list=index_list
            count = count + len(temp_list)
            print(count)
            thread_list=[]
            for i in range(len(temp_list)):
                d1=self.train[self.train.index == temp_list[i]]

                t=threading.Thread(target=KNNFeature.kNNTrain,name='thread{}'.format(i),args=(self,temp_list[i],neighbors,
                                                                                              amount_column,d1,feature_set,output,
                                                                                              pk_feature))
                thread_list.append(t)
                t.start()
                time.sleep(0.025)
            index_list=[x for x in index_list if x not in temp_list]
            for t in thread_list:
                t.join()

        print("train complete")

        print(self.train['normalize_amount'].isnull().sum())
        print(self.train['knn_b_value'].isnull().sum())
        print(self.train['knn_history'].isnull().sum())

        print(self.test['normalize_amount'].isnull().sum())
        print(self.test['knn_b_value'].isnull().sum())
        print(self.test['knn_history'].isnull().sum())

        self.train.fillna(0,inplace=True)
        self.test.fillna(0,inplace=True)
        a=['knn_history','knn_b_value','normalize_amount']
        print(a)
        return self.train,self.test,a
    #
    #     for i in index_list:
    #         d1=train[train.index == i]
    #         clusterset=KNNFeature.findCluster(self,neighbors,i,d1,feature_set,train)
    #         train.loc[train.index==i,'normalize_amount']=KNNFeature.normalizeAmount(self,clusterset,amount_column,d1[amount_column])
    #         train.loc[train.index==i,'knn_history']=KNNFeature.knnHistory(self,clusterset,output,pk_feature)
    #         train.loc[train.index==i,'knn_b_value']=KNNFeature.knnB_Value(self,clusterset,amount_column,output,d1[amount_column])
    #     train['knn_b_value'].replace(np.inf,0,inplace=True)
    #     print("train done!")
    #     for i in index_list1:
    #         d1=test[test.index==i]
    #         clusterset=KNNFeature.findCluster(neighbors,i,d1,feature_set,train)
    #         test.loc[test.index==i,'normalize_amount']=KNNFeature.normalizeAmount(self,clusterset,amount_column,d1[amount_column])
    #         test.loc[test.index == i, 'knn_history'] = KNNFeature.knnHistory(self,clusterset, output, pk_feature)
    #         test.loc[test.index == i, 'knn_b_value'] = KNNFeature.knnB_Value(self,clusterset, amount_column, output,d1[amount_column])
    #     test['knn_b_value'].replace(np.inf,0,inplace=True)
    #     print("test done!")
    #     return train,test

