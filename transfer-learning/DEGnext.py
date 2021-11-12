#DEGnext
#Copyright Tulika Kakati 2020
#Distributed under GPL v3 open source license
#This code may be used to test the DEGNextModel.pth on new datasets.

############################################

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import CancerDataset_global_pooled_multi_datasets_all
from sklearn.metrics import matthews_corrcoef
import DatasetTransform_global_pooled_all
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
dir_path = os.path.dirname(os.path.abspath(os.curdir))
#Input dataset
Tr_CancerProjects=["TCGA-BRCA","TCGA-KIRC","TCGA-KIRP","TCGA-LIHC","TCGA-LUAD","TCGA-LUSC","TCGA-PRAD","TCGA-THCA","TCGA-UCEC"] ###Training and fine-tuning datasets
Te_CancerProjects=["TCGA-BLCA","TCGA-CHOL","TCGA-COAD","TCGA-ESCA","TCGA-HNSC","TCGA-KICH","TCGA-READ","TCGA-STAD"] ###Testing datasets


#parameters
epoch=31
learning_rate=1e-4
np.random.seed(0)
torch.manual_seed(0)
loss_func = nn.BCEWithLogitsLoss()
loss_func_1 = nn.CrossEntropyLoss()
performance_by_dataset=[]
auc_file=[]
##CNN Architecture
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3,3), stride=1,padding=1),
            nn.Conv2d(6, 12, kernel_size=(3,3), stride=1,padding=1),
            #nn.MaxPool2d(2),
            nn.ReLU()
            )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=(3,3), stride=1,padding=1),
            nn.Conv2d(24, 48, kernel_size=(3,3), stride=1,padding=1),
            #nn.MaxPool2d(2),
            nn.ReLU()
            )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=(3,3), stride=1,padding=1),
            nn.Conv2d(64, 102, kernel_size=(3,3), stride=1,padding=1),
            #nn.MaxPool2d(2),
            nn.ReLU()
            )
        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(102, 164, kernel_size=(3,3), stride=1,padding=1),
            nn.Conv2d(164, 256, kernel_size=(3,3), stride=1,padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
            )
        self.global_pool = nn.AdaptiveMaxPool2d((1,1))
        self.Lin1 = nn.Sequential(
                nn.Linear(256,120),
                nn.ReLU()
            )
        self.Lin2 = nn.Sequential(
                nn.Linear(120,64),
                nn.ReLU()
            )
        self.Lin3 = nn.Sequential(
                nn.Linear(64,32),
                nn.ReLU()
                    )
        self.Lin4 = nn.Sequential(
                nn.Linear(32,8),
                nn.ReLU()
                    )
        self.Lin5 = nn.Sequential(
                nn.Linear(8,3),
                nn.ReLU()
                    )
        self.func=nn.Softmax(dim=1)
    def forward(self,x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.Lin1(x)
        x = self.Lin2(x)
        x = self.Lin3(x)
        x = self.Lin4(x)
        x = self.Lin5(x)
        output = self.func(x)
        return output



def calculate_accuracy_from_cm(Y_t,y_hat_class):
    cm_accuracy=accuracy_score(Y_t, y_hat_class,normalize=True)
    cm_recall=recall_score(Y_t, y_hat_class, average='weighted')
    precision=precision_score(Y_t, y_hat_class, average='weighted')
    cm_f_measure=f1_score(Y_t, y_hat_class, average='weighted')
    cm_mcc=matthews_corrcoef(Y_t,y_hat_class)
    return cm_accuracy*100, cm_recall*100, precision*100, cm_f_measure,cm_mcc


def train(model,samples):
    training_accuracy = []
    training_recall = []
    training_precision = []
    training_fmeasure = []
    training_mcc=[]
    training_loss=[]


    print("Training started....")
    for data in train_loader:
        img, y_img = data
        img = Variable(img)
        y_img = y_img.view(y_img.size(0), -1)
        y_img = Variable(y_img)
        X_t = torch.FloatTensor(img)
        y_hat= model(X_t)
        id=torch.arange(0,y_hat.shape[0])
        varr=y_img.flatten().long()
        true_label = torch.zeros(y_hat.shape[0],3)
        true_label[id,varr]=1
        loss = loss_func_1(y_hat,varr)
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
        optimizer.zero_grad()
        _,predicted=torch.max(y_hat,1)
        y_hat_class = predicted
        cm_accuracy, cm_recall, precision, f_measure,mcc=calculate_accuracy_from_cm(y_img.numpy(),y_hat_class)
        training_accuracy.append(cm_accuracy)
        training_recall.append(cm_recall)
        training_precision.append(precision)
        training_fmeasure.append(f_measure)
        training_loss.append(loss)
        training_mcc.append(mcc)

    return training_accuracy, training_recall,training_precision,training_fmeasure,training_mcc,training_loss


def fine_tune_train(model,samples):
    fine_tune_accuracy = []
    fine_tune_recall = []
    fine_tune_precision = []
    fine_tune_fmeasure = []
    fine_tune_mcc=[]
    fine_tune_loss=[]

    print("Fine tuning started....")
    for data in fine_tune_loader:
        img, y_img = data
        img = Variable(img)
        y_img = y_img.view(y_img.size(0), -1)
        y_img = Variable(y_img)
        X_t = torch.FloatTensor(img)
        y_hat= model(X_t)
        Y_t = torch.from_numpy(y_img.numpy())
        loss = loss_func(y_hat,Y_t)
        loss.backward()
        optimizer.step()
        fine_tune_loss.append(loss.item())
        optimizer.zero_grad()
        y_hat_class = np.where(y_hat<0.5, 0, 1)
        cm_accuracy, cm_recall, precision, fmeasure,mcc =calculate_accuracy_from_cm(y_img.numpy(),y_hat_class)
        fine_tune_accuracy.append(cm_accuracy)
        fine_tune_recall.append(cm_recall)
        fine_tune_precision.append(precision)
        fine_tune_fmeasure.append(fmeasure)
        fine_tune_loss.append(loss)
        fine_tune_mcc.append(mcc)

    accuracy=sum(fine_tune_accuracy)/len(fine_tune_accuracy)
    recall=sum(fine_tune_recall)/len(fine_tune_recall)
    precision=sum(fine_tune_precision)/len(fine_tune_precision)
    fmeasure=sum(fine_tune_fmeasure)/len(fine_tune_fmeasure)
    mcc=sum(fine_tune_mcc)/len(fine_tune_mcc)
    loss = sum(fine_tune_loss)/len(fine_tune_loss)

    return accuracy, recall,precision,fmeasure,mcc,loss



@torch.no_grad()
def bio_test(model,Project):
    test_loss = []
    test_accuracy = []
    test_recall = []
    test_precision = []
    test_fmeasure = []
    test_MCC=[]
    cum_accuracy=0
    cum_precision=0
    cum_MCC=0
    labels_predict=np.array([])
    prob_predict=np.array([])
    for data in test_loader:
        img, y_img = data
        img = Variable(img)
        y_img = y_img.view(y_img.size(0), -1)
        y_img = Variable(y_img)
        X_t = torch.FloatTensor(img)
        y_hat= model(X_t)
        Y_t = torch.from_numpy(y_img.numpy())
        loss = loss_func(y_hat,Y_t)
        test_loss.append(loss.item())
        y_hat_class = np.where(y_hat<0.5, 0, 1)
        labels_predict=np.append(labels_predict,y_hat_class)
        prob_predict=np.append(prob_predict,y_hat)
        cm_accuracy, cm_recall, precision, fmeasure,mcc =calculate_accuracy_from_cm(y_img.numpy(),y_hat_class)
        test_accuracy.append(cm_accuracy)
        test_recall.append(cm_recall)
        test_precision.append(precision)
        test_fmeasure.append(fmeasure)
        test_MCC.append(mcc)
    File = 'X_data_test_predicted_labels_'+str(Project)+'.txt'
    labels_predict=labels_predict.astype(np.int)
    np.savetxt(File, labels_predict,fmt='%i')
    File = 'X_data_test_predicted_probs_'+str(Project)+'.txt'
    np.savetxt(File, prob_predict)
    cum_accuracy=sum(test_accuracy)/len(test_accuracy)
    cum_recall=sum(test_recall)/len(test_recall)
    cum_precision=sum(test_precision)/len(test_precision)
    cum_fmeasure=sum(test_fmeasure)/len(test_fmeasure)
    cum_MCC=sum(test_MCC)/len(test_MCC)
    return cum_accuracy, cum_recall,cum_precision,cum_fmeasure,cum_MCC

##Network initialize

model = MyNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
performance_by_dataset=[]
testing_accuracy = []
testing_recall = []
testing_precision = []
testing_fmeasure = []
testing_mcc=[]
testing_loss=[]

######################Training DEGnext model on T1 data of the training datasets ######################
for i in range(0,epoch):
    if(i>1):
        MyModelName='DEGNextModel.pth'
        model.load_state_dict(torch.load(MyModelName))
        model.eval()
        print('Model loaded...')
    for j in range(0,len(Tr_CancerProjects)):
        Project=Tr_CancerProjects[j]
        print("Training for ",Project," started....")
        train_dataset = CancerDataset_global_pooled_multi_datasets_all.MyDataset(data_type=Project,case='train',transform=DatasetTransform_global_pooled_all.Compose([DatasetTransform_global_pooled_all.ArrayToTensor()]))
        batch_size = 256
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        samples=train_dataset.len
        training_accuracy,training_recall, training_precision,training_fmeasure,training_mcc,training_loss=train(model,samples)
        accuracy=sum(training_accuracy)/len(training_accuracy)
        recall=sum(training_recall)/len(training_recall)
        precision=sum(training_precision)/len(training_precision)
        fmeasure=sum(training_fmeasure)/len(training_fmeasure)
        mcc=sum(training_mcc)/len(training_mcc)
        loss = sum(training_loss)/len(training_loss)
        Train_results='Epoch '+str(i)+' Train results for '+Project+': accuracy: '+str(accuracy)+' recall: '+str(recall)+'  precision: '+str(precision)+' fmeasure: '+str(fmeasure)+' mcc: '+str(mcc)+' loss: '+str(loss)
        print(Train_results)
        performance_by_dataset=np.append(performance_by_dataset,Train_results)

    cum_accuracy=sum(training_accuracy)/len(training_accuracy)
    cum_recall=sum(training_recall)/len(training_recall)
    cum_precision=sum(training_precision)/len(training_precision)
    cum_fmeasure=sum(training_fmeasure)/len(training_fmeasure)
    cum_loss = sum(training_loss)/len(training_loss)
    cum_mcc=sum(training_mcc)/len(training_mcc)
    print("average training evaluation for all datasets for epoch ",str(i)," :",cum_loss, cum_accuracy, cum_recall,cum_precision,cum_fmeasure,cum_mcc)
    File='Train_accuracy_selected_datasets_continued.txt'
    np.savetxt(File, performance_by_dataset, fmt='%s')
    MyModelName='DEGNextModel.pth'
    torch.save(model.state_dict(), MyModelName)



print("Training ended....")




##########Fine-tuning of the DEGnext model with the fine tune F1 data of training datasets ########################
model = MyNetwork()
MyModelName='DEGNextModel.pth'
model.Lin5=nn.Linear(8,1)
model.func=nn.Identity()
model.load_state_dict(torch.load(MyModelName))
model.eval()
print('Model loaded...')
performance_by_dataset=[]
performance_test=[]

for i in range(0,epoch):
    fine_tune_accuracy_1 = []
    fine_tune_recall_1 = []
    fine_tune_precision_1 = []
    fine_tune_fmeasure_1 = []
    fine_tune_mcc_1=[]
    fine_tune_loss_1=[]

    for j in range(0,len(Tr_CancerProjects)):

        Project=Tr_CancerProjects[j]
        print("Fine tuning for ",Project," started....")
        fine_tune_dataset = CancerDataset_global_pooled_multi_datasets_all.MyDataset(data_type=Project,case='bio-fine-tune',transform=DatasetTransform_global_pooled_all.Compose([DatasetTransform_global_pooled_all.ArrayToTensor()]))
        batch_size = 64
        fine_tune_loader = DataLoader(dataset=fine_tune_dataset, batch_size=batch_size, shuffle=True)
        samples=fine_tune_dataset.len
        accuracy,recall, precision,fmeasure,mcc,loss=fine_tune_train(model,samples)
        Fine_tune_results='Epoch '+str(i)+' Fine tune results for '+Project+': accuarcy: '+str(accuracy)+' recall:  '+str(recall)+' precision: '+str(precision)+' fmeasure: '+str(fmeasure)+' mcc:  '+str(mcc)+' loss: '+str(loss)
        print(Fine_tune_results)
        performance_by_dataset=np.append(performance_by_dataset,Fine_tune_results)
        fine_tune_accuracy_1.append(accuracy)
        fine_tune_recall_1.append(recall)
        fine_tune_precision_1.append(precision)
        fine_tune_fmeasure_1.append(fmeasure)
        fine_tune_loss_1.append(loss)
        fine_tune_mcc_1.append(mcc)



    cum_accuracy=sum(fine_tune_accuracy_1)/len(fine_tune_accuracy_1)
    cum_recall=sum(fine_tune_recall_1)/len(fine_tune_recall_1)
    cum_precision=sum(fine_tune_precision_1)/len(fine_tune_precision_1)
    cum_fmeasure=sum(fine_tune_fmeasure_1)/len(fine_tune_fmeasure_1)
    cum_loss = sum(fine_tune_loss_1)/len(fine_tune_loss_1)
    cum_mcc=sum(fine_tune_mcc_1)/len(fine_tune_mcc_1)
    print("Fine tuning evaluation for all datasets epoch ",str(i)," :",cum_loss, cum_accuracy, cum_recall,cum_precision,cum_fmeasure,cum_mcc)
    File='Fine_tuning_accuracy_selected_datasets.txt'
    np.savetxt(File, performance_by_dataset, fmt='%s')

###############Testing DEGnext model on biologically validated data (T3+F1) data of the 3 untrained test datasets###################
performance_by_dataset=[]
for j in range(0,len(Te_CancerProjects)):
    Project=Te_CancerProjects[j]
    print("bio testing for ",Project," started....")
    test_dataset = CancerDataset_global_pooled_multi_datasets_all.MyDataset(data_type=Project,case='bio-data',transform=DatasetTransform_global_pooled_all.Compose([DatasetTransform_global_pooled_all.ArrayToTensor()]))
    batch_size = 64
    samples=test_dataset.len
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    accuracy,recall, precision,fmeasure,mcc=bio_test(model,Project)
    Test_results='Test results for '+Project+': accuarcy: '+str(accuracy)+' recall: '+str(recall)+' precision: '+str(precision)+' fmeasure: '+str(fmeasure)+' mcc: '+str(mcc)
    print(Test_results)
    performance_by_dataset=np.append(performance_by_dataset,Test_results)

File='Bio_test_data_accuracy_selected_datasets.txt'
np.savetxt(File, performance_by_dataset, fmt='%s')
print("Testing ended....")

File = 'X_data_test_predicted_labels_'+str(Project)+'.txt'
xy_predicted_labels = np.loadtxt(File,delimiter=',', dtype=np.float32)
xy_predicted_labels=np.where(xy_predicted_labels=="1.","UR", xy_predicted_labels)
xy_predicted_labels=np.where(xy_predicted_labels=="1.0","--UR", xy_predicted_labels)
xy_predicted_labels=np.where(xy_predicted_labels=="0.0","--DR", xy_predicted_labels)

File = dir_path+'/datasets/'+Project+'_biologically_validated_genes.csv'
genes = pandas.read_csv(File,header=None)
df=genes.values.flatten()
rows=df.shape[0]
df=np.resize(df,rows).reshape(rows,1)
xy_predicted_labels=np.resize(xy_predicted_labels,rows).reshape(rows,1)

classified_genes=np.concatenate((df,xy_predicted_labels),axis=1)
File='Predicted_UR_and_DR_genes.txt'
np.savetxt(File, classified_genes, fmt='%s')


