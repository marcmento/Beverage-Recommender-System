## COMMENT FOR MARKER
## THIS PROGRAM CAN TAKE UP TO 15MINS TO RUN
## MY LAPTOP MAY JUST BE SLOW BUT THIS PROGRAM WOULD TAK ME ANYWHERE
## FROM 10-15MINS TO RUN EVERY TIME


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD, NMF, accuracy, KNNBaseline
from surprise.model_selection import KFold, train_test_split, GridSearchCV
from lightfm import LightFM
from lightfm_dataset_helper.lightfm_dataset_helper import DatasetHelper
import scipy.interpolate
from surprise import AlgoBase

def SVD_Search():
    print("SVD GRID SEARCH BEGIN....")
    data1 = Dataset.load_from_df(cleant[['BeerID','ReviewerID', 'rating']],reader)
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.4, 0.6]}
    gs_svd = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs_svd.fit(data1)
    print(gs_svd.best_score['mae'])
    print(gs_svd.best_params['mae'])

def KNN_Search():
    print("KNN GRID SEARCH BEGIN....")
    data1 = Dataset.load_from_df(cleant[['BeerID','ReviewerID', 'rating']],reader)
    param_grid = {'k': [10, 30, 40, 60, 80], 'sim_options': {'user_based': [True, False]},\
                'bsl_options': {'method': ['als', 'sgd']}}
    gs_knn = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)
    gs_knn.fit(data1)
    print(gs_knn.best_score['mae'])
    print(gs_knn.best_params['mae'])

def NMF_Search():
    print("NMF GRID SEARCH BEGIN....")
    data1 = Dataset.load_from_df(cleant[['BeerID','ReviewerID', 'rating']],reader)
    param_grid = {'n_epochs': [10, 50, 90], 'n_factors': [10,15,20], 'biased': [True, False]}
    gsNMF = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=3)
    gsNMF.fit(data1)
    print(gsNMF.best_score['mae'])
    print(gsNMF.best_params['mae'])

def Basic_Algos_Ensemble():
    print("STARTING BASIC ALGORITHMS....")
    #SVD
    svd = SVD()
    modelSVD = svd.fit(trainset)
    predictionssvd = svd.test(valset)
    print("SVD model on validation set = " + str(accuracy.mae(predictionssvd,verbose=True)))
    #NMF
    nmf = NMF(n_epochs =  90, n_factors = 10, biased = True)
    modelNMF = nmf.fit(trainset)
    predictionsnmf = nmf.test(valset)    
    print("NMF model on validation set = " + str(accuracy.mae(predictionsnmf,verbose=True)))
    #KNN
    knnb = KNNBaseline(k=80, sim_options = {'user_based': False}, bsl_options = {'method': 'als'})
    modelKNN = knnb.fit(trainset)
    predictionsknn = knnb.test(valset)
    print("KNN model on validation set = " + str(accuracy.mae(predictionsknn,verbose=True)))
    #Test set predicitons 
    print("STARTING PREDICTION ON TEST SET")
    UID = df_test['ReviewerID']
    IID = df_test['BeerID']
    ROWIDS = df_test['RowID']
    #Writing to TSV files
    i = 0
    SVDpred = []
    while i < len(UID):
        predhold = modelSVD.predict(UID[i],IID[i])
        SVDpred.append(predhold)
        i = i+1
    with open('A3-1.tsv', 'w', newline='') as outf:
        i = 0
        while i < len(UID):
            getpred = SVDpred[i]
            outf.write(str(ROWIDS[i]) + '\t' + str(getpred[3]) + '\n')
            i += 1
    print("A3-1 CREATED")        
    i = 0
    NMFpred = []
    while i < len(UID):
        predhold = modelNMF.predict(UID[i],IID[i])
        NMFpred.append(predhold)
        i = i+1
    with open('A3-2.tsv', 'w', newline='') as outf:
        i = 0
        while i < len(UID):
            getpred = NMFpred[i]
            outf.write(str(ROWIDS[i]) + '\t' + str(getpred[3]) + '\n')
            i += 1
    print("A3-2 CREATED")        
    i = 0
    KNNpred = []
    while i < len(UID):
        predhold = modelKNN.predict(UID[i],IID[i])
        KNNpred.append(predhold)
        i = i+1
    with open('A3-3.tsv', 'w', newline='') as outf:
        i = 0
        while i < len(UID):
            getpred = KNNpred[i]
            outf.write(str(ROWIDS[i]) + '\t' + str(getpred[3]) + '\n')
            i += 1
    print("A3-3 CREATED")        
    print("ALL BASIC ALGORITHMS DONE")        
    svdEns = SVD()
    nmfEns = NMF(n_factors = 10, biased = True)
    knnbEns = KNNBaseline(k=80, sim_options = {'user_based': False}, bsl_options = {'method': 'als'})
    print("STARTING ENSEMBLE")
    Hybrid = HybridAlgorithm([svdEns, nmfEns, knnbEns], [0.33, 0.33, 0.33])
    Hybrid.fit(trainset)         
    print("STARTING PREDICTION ON TEST SET")
    i = 0
    Hybridpred = []
    while i < len(UID):
        predhold = Hybrid.predict(UID[i],IID[i])
        Hybridpred.append(predhold)
        i = i+1
    with open('A3-4.tsv', 'w', newline='') as outf:
        i = 0
        while i < len(UID):
            getpred = Hybridpred[i]
            outf.write(str(ROWIDS[i]) + '\t' + str(getpred[3]) + '\n')
            i += 1
    print("A3-4 CREATED")   


class HybridAlgorithm(AlgoBase):

    def __init__(self, algorithms, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
        self.weights = weights

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        for algorithm in self.algorithms:
            algorithm.fit(trainset)
                
        return self

    def estimate(self, u, i):
        
        sumScores = 0
        sumWeights = 0
        
        for idx in range(len(self.algorithms)):
            if type(self.algorithms[idx].estimate(u, i)) == tuple :
                sumScores += self.algorithms[idx].estimate(u, i)[0] * self.weights[idx]
            else:
                sumScores += self.algorithms[idx].estimate(u, i) * self.weights[idx]
            sumWeights += self.weights[idx]

        return sumScores / sumWeights

def Advanced_Model():
    print("ADVANCED MODEL STARTING....")       
    #Getting features and mapping them to row ID for both users and items
    col_list = ["RowID", "Gender"]
    userdf = pd.read_csv("features.tsv", usecols=col_list, sep='\t', names=['RowID','BrewerID','ABV','DayofWeek','Month',
                                                                 'DayofMonth','Year','TimeOfDay','Gender',
                                                                 'Birthday','Text','Lemmatized','POS_Tag'])
    cleanup_nums = {"Gender":     {"Male": 1, "Female": 2, "unknown": 0}}
    userdf = userdf.replace(cleanup_nums)
    userfeature = pd.merge(userdf, df_train, on=["RowID"])
    userfeature = userfeature.drop(['RowID', 'BeerID', 'BeerName', 'BeerType', 'rating'],axis=1)
    col_list = ["RowID", "BrewerID", "ABV"]
    itemdf = pd.read_csv("features.tsv", usecols=col_list, sep='\t', names=['RowID','BrewerID','ABV','DayofWeek','Month',
                                                                 'DayofMonth','Year','TimeOfDay','Gender',
                                                                 'Birthday','Text','Lemmatized','POS_Tag'])
    items = df_train.drop(['ReviewerID','rating'],axis=1)
    itemsMerged = pd.merge(itemdf, items, on=["RowID"])
    itemsdict = itemsMerged.drop(['RowID'],axis=1)
    #Set up for dataset helper
    items_column = "BeerID"
    user_column = "ReviewerID"
    ratings_column = "rating"
    items_feature_columns = ["BrewerID", "ABV", "BeerName", "BeerType"]
    user_features_columns = ["Gender"]      
    print("DATASET HELPER STARTING....")              
    dataset_helper_instance = DatasetHelper(
        users_dataframe=userfeature,
        items_dataframe=itemsdict,
        interactions_dataframe=cleant,
        item_id_column=items_column,
        items_feature_columns=items_feature_columns,
        user_id_column=user_column,
        user_features_columns=user_features_columns,
        interaction_column=ratings_column,
        clean_unknown_interactions=True,
    )
    dataset_helper_instance.routine()    
    #Creating model
    print("CREATING MODEL....")       
    model = LightFM(no_components=24, loss="warp", k=15)
    model.fit(
        interactions=dataset_helper_instance.interactions,
        sample_weight=dataset_helper_instance.weights,
        item_features=dataset_helper_instance.item_features_list,
        user_features=dataset_helper_instance.user_features_list,
        verbose=True,
        epochs=10,
        num_threads=20,
    )
    # Making predictions on train
    usertrain = df_train['ReviewerID']
    itemtrain = df_train['BeerID']
    usertrain = usertrain.to_numpy()
    itemtrain = itemtrain.to_numpy()
    trainRatings = df_train['rating']
    predictionstrain = model.predict(user_ids = usertrain, item_ids = itemtrain)
    #Creating refernece  
    y_interp = scipy.interpolate.interp1d(predictionstrain, trainRatings)
    #Setting up test and making predicitons on it
    cleantest = df_test.drop(['RowID','BeerName','BeerType'],axis=1)
    usertest = cleantest['ReviewerID']
    itemtest = cleantest['BeerID']
    usertest = usertest.to_numpy()
    itemtest = itemtest.to_numpy()
    RowID = df_test['RowID']
    predictionstest = model.predict(user_ids = usertest, item_ids = itemtest)
    print("CONVERTING IMPLICIT SCORES....")       
    i = 0
    preds = []
    minpred =  min(predictionstrain)
    while i < len(predictionstest):
        if predictionstest[i] < minpred:
            predictionstest[i] = minpred
        holder = y_interp(predictionstest[i])
        preds.append(holder)
        i = i + 1     
    with open('A3-5.tsv', 'w', newline='') as outf:
        i = 0
        while i < len(preds):
            outf.write(str(RowID[i]) + '\t' + str(preds[i]) + '\n')
            i += 1
    print("A3-5 CREATED")  
    print("ADVANCED MODEL FINISHED")  

# Main
if __name__ == "__main__":
    print("LOADING IN DATA...")  
    df_train = pd.read_csv('train.tsv',sep='\t', names=['RowID','BeerID','ReviewerID','BeerName','BeerType','rating'])
    df_val = pd.read_csv('val.tsv',sep='\t', names=['RowID','BeerID','ReviewerID', 'BeerName','BeerType','rating'])
    df_test = pd.read_csv('test.tsv',sep='\t', names=['RowID','BeerID','ReviewerID','BeerName','BeerType'])
    cleant = df_train.drop(['RowID','BeerName','BeerType'],axis=1)
    cleanv = df_val.drop(['RowID','BeerName','BeerType'],axis=1)
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(cleant[['BeerID','ReviewerID', 'rating']],reader)
    datav = Dataset.load_from_df(cleanv[['BeerID','ReviewerID', 'rating']],reader)
    trainset = data.build_full_trainset()
    NA , valset = train_test_split(datav, test_size=1.0)

    sweep = False

    if sweep == True:
        SVD_Search()
        KNN_Search()
        NMF_Search()
    else: 
        Basic_Algos_Ensemble()
        Advanced_Model()

