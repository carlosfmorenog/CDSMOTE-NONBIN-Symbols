#%% Code of the CDSMOTE method modified to work with multiclass datasets
 
# Please reference the following paper:
    
# Elyan E., Moreno-García C.F., Jayne C., CDSMOTE: class decomposition and synthetic minority class oversampling technique for imbalanced-data classification. Neural Comput Appl. 2020. doi:10.1007/s00521-020-05130-z
# @article{Elyan2020,
# author = {Elyan, Eyad and Moreno-Garcia, Carlos Francisco and Jayne, Chrisina},
# doi = {10.1007/s00521-020-05130-z},
# isbn = {0123456789},
# issn = {1433-3058},
# journal = {Neural Computing and Applications},
# publisher = {Springer London},
# title = {{CDSMOTE: class decomposition and synthetic minority class oversampling technique for imbalanced-data classification}},
# url = {https://doi.org/10.1007/s00521-020-05130-z},
# year = {2020}
# }

#%% 0. Import necessary packages

import sys
sys.path.append('/utils')
import clustData
import computeKVs
import numpy as np
import csv
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn import svm

#%% ############################# 1. Input params #############################

dataset = 'symbols_combined_pixel_red' # name of the csv file containing the data and target
classdecomp = 'Kmeans' # 'FCmeans', 'FCmeansOptimised' and 'DBSCAN' also available 
oversampler = 'SMOTE' #'ADASYN' also available
threshold = 10 # if samples in positive class are apart from average by more than this value, apply oversampling (Sec 3.2 paper)
k_type = 'fixed' # Indicates how to calculate k values for class decomposition
# Choose from:
# "fixed": The majority class is decomposed using k=n_clusters
# "ir": The majority class is decomposed using k=ceil(IR), where IR is the imbalance ratio between the majority and the minority class. THIS RARELY LEADS TO OVERSAMPLING BEING USED!
n_clusters = 10 # used in option "majority"
# number_of_tests = 10 # How many times to repeat the SVM experiment comparing the original and new db

#%% ############################## 2. Load data ###############################

## 1. Load dataset
with open('data//'+str(dataset)+'.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = []
    target = []
    for row in reader:
        data.append(list(map(float,row[0:len(row)-1])))
        target.append(row[-1])
del row, reader, f

## 2. Find majority and minority classes
majority_class = max(set(target), key=target.count)
minority_class = min(set(target), key=target.count)

## 3. Plot distribution of original dataset
print('Dataset: '+str(dataset))
histo = [['Class','Number of Samples']]
for i, label1 in enumerate(sorted(list(set(target)))):
    cont = 0
    for j, label2 in enumerate(target):
        if label1 == label2:
            cont+=1
    histo.append([label1,cont])
histo.append(['Total Samples', len(target)])
# Save the histogram as a .csv file   
with open('results//originaldb_classdistribution.csv', 'w', newline='', encoding='utf-8') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    for i, hist in enumerate(histo):
        filewriter.writerow(hist)
# Load as a panda
histo_panda = pd.DataFrame.from_records(histo[1:-1], columns=histo[0])
print(histo_panda)
print('Total samples: '+str(len(target)))
# Create a histogram using seaborn
sns_plot = sns.barplot(x="Class", y="Number of Samples", data=histo_panda)
# Save the image
sns_plot.figure.set_size_inches(10, 6)
sns_plot.figure.savefig('results//originaldb_barchart'+'.jpg', orientation = 'landscape', dpi = 600)
print('\nShowing class distribution bar chart...')
plt.show()
    
#%% ######################### 3. Class decomposition #########################

## 1. Calculate k vector (for class decomposition)

if k_type.lower() == 'fixed':
    k = computeKVs.majority(data, target, n_clusters)
elif k_type.lower() == 'ir':
    ## Calculate the IR between the majority and the minority
    majority_samples = histo_panda.loc[histo_panda['Class'] == majority_class].reset_index()
    minority_samples = histo_panda.loc[histo_panda['Class'] == minority_class].reset_index()
    n_clusters = math.ceil(majority_samples['Number of Samples'][0]/minority_samples['Number of Samples'][0])                                                                                                                                                                   
    k = computeKVs.majority(data, target, n_clusters)
else:
    print('Invalid k values selecting option for CDSMOTE')
    sys.exit()

## 2. Cluster the data
if classdecomp.lower()=='kmeans':
    target_cd = clustData.Kmeans(data, target, k)
elif classdecomp.lower()=='fcmeans':
    target_cd = clustData.FCmeans(data, target, k)
elif classdecomp.lower()=='fcmeansoptimised':
     target_cd = clustData.FCmeansOptimised(data, target, k, max_nclusters = 10)   
elif classdecomp.lower()=='dbscan':        
     target_cd = clustData.DBSCAN(data, target, k, eps=0.5, min_samples=5)
else:
    print('Invalid clustering algorithm selected.')
    sys.exit()
    
## 3. Plot distribution after cd
histo = [['Class','Number of Samples']]
for i, label1 in enumerate(sorted(list(set(target_cd)))):
    cont = 0
    for j, label2 in enumerate(target_cd):
        if label1 == label2:
            cont+=1
    histo.append([label1,cont])
histo.append(['Total Samples', len(target_cd)])
# Save the histogram as a .csv file   
with open('results//decomposeddb_classdistribution.csv', 'w', newline='', encoding='utf-8') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    for i, hist in enumerate(histo):
        filewriter.writerow(hist)
# Load as a panda
histo_panda = pd.DataFrame.from_records(histo[1:-1], columns=histo[0])
print(histo_panda)
print('Total samples: '+str(len(target_cd)))
# Create a histogram using seaborn
sns_plot = sns.barplot(x="Class", y="Number of Samples", data=histo_panda)
# draw a line depicting the average
indexesUnique = list(set(target_cd))
indexesUnique.sort()
cdclassdist_count = []
cdclassdist_names = []
for cdclass in indexesUnique:
     cdclassdist_count.append(target_cd.count(cdclass))
     cdclassdist_names.append(cdclass)
average = sum(cdclassdist_count)/len(cdclassdist_count)
print('Average number of samples per class:', average)
plt.axhline(average, color='red')
# Save the image
sns_plot.figure.set_size_inches(10, 6)
sns_plot.figure.savefig('results//decomposeddb_barchart'+'.jpg', orientation = 'landscape', dpi = 600)
print('\nShowing class distribution bar chart...')
plt.show()
     
#%% ############################ 4. Oversampling #############################


## 1. Calculate reference class (i.e. closest to the average and above it) for oversampling
c = np.inf
ref = majority_class+'_c0' # gets picked by default if none other accomplishes
for i,j in enumerate(cdclassdist_count):
    if abs(j-average)<c and j-average>=0:
        c = abs(j-average)
        ref = cdclassdist_names[i]

data_cdsmote = list(np.asarray(data)[(np.where(np.asarray(target)==majority_class))])
target_cdsmote = list(np.asarray(target_cd)[(np.where(np.asarray(target)==majority_class))])

## 2. For all non-majority classes (considering the original dataset), see if they are far (i.e. difference greater than the threshold) from the average (red line in the last plot)
flag = 0
for i, cdclassdist_name in enumerate(cdclassdist_names):
    if majority_class not in cdclassdist_name.split('_')[0]:
        if abs(average-cdclassdist_count[i])>threshold and average-cdclassdist_count[i]>=0:
            flag = 1
            print('Oversampling class '+str(cdclassdist_name)+'...')            
            ## 3. Create a sub-dataset that only contains the new majority and current non-minority classes
            data_majmin = []
            target_majmin = []
            for j, label in enumerate(target_cd):
                if label == cdclassdist_name or label == ref:
                    data_majmin.append(data[j])
                    target_majmin.append(label)
            ## 4. Do the oversampling
            if oversampler.lower() == 'smote':
                sm = SMOTE()
                data_over, target_over = sm.fit_resample(data_majmin, target_majmin) 
            elif oversampler.lower() == 'adasyn':
                ada = ADASYN()
                data_over, target_over = ada.fit_resample(data_majmin, target_majmin)
            else:
                print('Invalid oversampling algorithm.')
                sys.exit() 
            # Append the oversampled data to the new repository
            for j, label in enumerate(target_over):
                if label == cdclassdist_name:
                    data_cdsmote.append(list(data_over[j]))
                    target_cdsmote.append(label)
        else:
            # Append the not-oversampled
            for j, label in enumerate(target_cd):
                if label == cdclassdist_name:
                    data_cdsmote.append(list(data[j]))
                    target_cdsmote.append(label)
            
    
## 5. Plot distribution after smote
if flag == 1:
    histo = [['Class','Number of Samples']]
    for i, label1 in enumerate(sorted(list(set(target_cdsmote)))):
        cont = 0
        for j, label2 in enumerate(target_cdsmote):
            if label1 == label2:
                cont+=1
        histo.append([label1,cont])
    histo.append(['Total Samples', len(target_cdsmote)])
    ## Save the histogram as a .csv file   
    with open('results/cdsmotedb_classdistribution.csv', 'w', newline='', encoding='utf-8') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for i, hist in enumerate(histo):
            filewriter.writerow(hist)
    ## Load as a panda
    histo_panda = pd.DataFrame.from_records(histo[1:-1], columns=histo[0])
    print(histo_panda)
    print('Total samples: '+str(len(target_cdsmote)))
    ## Create a histogram using seaborn
    sns_plot = sns.barplot(x="Class", y="Number of Samples", data=histo_panda)
    ## draw a line depicting the average
    indexesUnique = list(set(target_cdsmote))
    indexesUnique.sort()
    newestclassdist_count = []
    for newestclass in indexesUnique:
          newestclassdist_count.append(target_cdsmote.count(newestclass))
    average_new = sum(newestclassdist_count)/len(newestclassdist_count)
    print('New average number of samples per class:', average_new)
    plt.axhline(average, color='red')
    plt.axhline(average_new, color='blue')
    ## Save the image
    sns_plot.figure.set_size_inches(10, 6)
    sns_plot.figure.savefig('results/cdsmotedb_barchart'+'.jpg', orientation = 'landscape', dpi = 600)
    print('\nShowing class distribution bar chart...')
    plt.show()
else:
    print('All non-majority classes are close to average. No oversampling was needed.')


## Save the new dataset
with open('results/cdsmotedb.csv', 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    for j, tar in enumerate(target_cdsmote):
        row = list(data_cdsmote[j].copy())
        row.append(target_cdsmote[j])
        filewriter.writerow(row)


# #%% ########################### 5. Classification ############################

# accuracy_o_final = 0
# accuracy_c_final = 0

# ## Split data/target and data_cdsmote/target_cdsmote (stratified and with many splits)
# sss = StratifiedShuffleSplit(n_splits=number_of_tests, test_size=0.3, random_state=42)

# # Original data
# experiments = 0
# for train_index, test_index in sss.split(data, target):
#     print('\nExperiment '+str(experiments)+' Original DB...')
#     experiments+=1
#     X_train_o, X_test_o,  = np.asarray(data)[train_index], np.asarray(data)[test_index]
#     y_train_o, y_test_o = np.asarray(target)[train_index], np.asarray(target)[test_index]
#     # Train SVM models
#     clf_o = svm.SVC(kernel='linear')
#     clf_o.fit(X_train_o, y_train_o)
#     y_pred_o = clf_o.predict(X_test_o)
#     # Test 
#     from sklearn import metrics
#     print("Accuracy Original DB:",metrics.accuracy_score(y_test_o, y_pred_o))
#     accuracy_o_final = accuracy_o_final + metrics.accuracy_score(y_test_o, y_pred_o)
    
# # CDSMOTE data
# experiments = 0
# for train_index, test_index in sss.split(data_cdsmote, target_cdsmote):
#     print('\nExperiment '+str(experiments)+' CDSMOTE DB...')
#     experiments+=1
#     X_train_c, X_test_c,  = np.asarray(data_cdsmote)[train_index], np.asarray(data_cdsmote)[test_index]
#     y_train_c, y_test_c = np.asarray(target_cdsmote)[train_index], np.asarray(target_cdsmote)[test_index]
#     # Train SVM models
#     clf_c = svm.SVC(kernel='linear')
#     clf_c.fit(X_train_c, y_train_c)
#     y_pred_c = clf_c.predict(X_test_c)
#     # Test, making sure accuracy considers sub_classes as good
#     accuracy_c=0
#     for i,label in enumerate(y_test_c):
#         if label.split('_')[0] == y_pred_c[i].split('_')[0]:
#            accuracy_c+=1
#     print("Accuracy CDSMOTE DB:",accuracy_c/len(y_pred_c))
#     accuracy_c_final = accuracy_c_final + accuracy_c/len(y_pred_c)
    
# # Final results
# print('\nFinal Results:')
# print("Average Accuracy Original DB:",accuracy_o_final/experiments)
# print("Average Accuracy CDSMOTE DB:",accuracy_c_final/experiments)
