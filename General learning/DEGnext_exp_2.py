#DEGnext 
#Copyright Tulika Kakati 2020 
#Distributed under GPL v3 open source license
#This code may be used to fine-tune and test the model on F1 and T3 data for all the cancer datatsets and save the model for each cancer dataset.
##################################
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

epoch=30
learning_rate=1e-4
np.random.seed(0)
torch.manual_seed(0)
loss_func = nn.BCEWithLogitsLoss()

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3,3), stride=1,padding=1),
            nn.Conv2d(6, 12, kernel_size=(3,3), stride=1,padding=1),
            nn.ReLU()
            )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=(3,3), stride=1,padding=1),
            nn.Conv2d(24, 48, kernel_size=(3,3), stride=1,padding=1),
            nn.ReLU()
            )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=(3,3), stride=1,padding=1),
            nn.Conv2d(64, 102, kernel_size=(3,3), stride=1,padding=1),
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
		#####Change in the output layer since labels for fine-tune data F1 and bio-test data T3 have two labels
        #m = nn.Softmax(dim=1)
        #output = m(x)
        output = x
        return output


def calculate_accuracy_from_cm(Y_t,y_hat_class):
    cm_accuracy=accuracy_score(Y_t, y_hat_class,normalize=True)
    cm_recall=recall_score(Y_t, y_hat_class, average='weighted')
    precision=precision_score(Y_t, y_hat_class, average='weighted')
    cm_f_measure=f1_score(Y_t, y_hat_class, average='weighted')
    cm_mcc=matthews_corrcoef(Y_t,y_hat_class)
    return cm_accuracy*100, cm_recall*100, precision*100, cm_f_measure,cm_mcc

def fine_tune_train(model,samples):
    fine_tune_loss = []
    cum_accuracy=0
    cum_precision=0
    for i in range(epoch):
        training_loss = []
        fine_tune_accuracy = []
        fine_tune_recall = []
        fine_tune_precision = []
        fine_tune_fmeasure = []
        fine_tune_mcc=[]
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
            training_loss.append(loss.item())
            optimizer.zero_grad()
            y_hat_class = np.where(y_hat<0.5, 0, 1)
            cm_accuracy, cm_recall, precision, fmeasure,mcc =calculate_accuracy_from_cm(y_img.numpy(),y_hat_class)
            fine_tune_accuracy.append(cm_accuracy)
            fine_tune_recall.append(cm_recall)
            fine_tune_precision.append(precision)
            fine_tune_fmeasure.append(fmeasure)
            fine_tune_loss.append(loss)
            fine_tune_mcc.append(mcc)
        cum_accuracy=sum(fine_tune_accuracy)/len(fine_tune_accuracy)
        cum_recall=sum(fine_tune_recall)/len(fine_tune_recall)
        cum_precision=sum(fine_tune_precision)/len(fine_tune_precision)
        cum_fmeasure=sum(fine_tune_fmeasure)/len(fine_tune_fmeasure)
        cum_loss=sum(fine_tune_loss)/len(fine_tune_loss)
        cum_mcc=sum(fine_tune_mcc)/len(fine_tune_mcc)
    return cum_accuracy, cum_recall,cum_precision,cum_fmeasure,cum_mcc

@torch.no_grad()
def test(model,Project):
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
        mcc=matthews_corrcoef(y_hat_class,y_img.numpy())
        cm_accuracy, cm_recall, precision, fmeasure,mcc =calculate_accuracy_from_cm(y_img.numpy(),y_hat_class)
        test_accuracy.append(cm_accuracy)
        test_recall.append(cm_recall)
        test_precision.append(precision)
        test_fmeasure.append(fmeasure)
        test_MCC.append(mcc)
    File = 'Test_predicted_labels_'+str(Project)+'.txt'
    labels_predict=labels_predict.astype(np.int)
    np.savetxt(File, labels_predict,fmt='%i')
    File = 'Test_predicted_probs_'+str(Project)+'.txt'
    np.savetxt(File, prob_predict)
    cum_accuracy=sum(test_accuracy)/len(test_accuracy)
    cum_recall=sum(test_recall)/len(test_recall)
    cum_precision=sum(test_precision)/len(test_precision)
    cum_fmeasure=sum(test_fmeasure)/len(test_fmeasure)
    cum_MCC=sum(test_MCC)/len(test_MCC)
    return cum_accuracy, cum_recall,cum_precision,cum_fmeasure,cum_MCC

CancerProjects=["TCGA-CHOL","TCGA-COAD","TCGA-ESCA","TCGA-HNSC","TCGA-KICH"]
performance_by_dataset=[]
auc_file=[]

for i in range(0,len(CancerProjects)):
    Project=CancerProjects[i]
    print("Fine tuning and testing for ",Project," started....")
    performance_by_dataset=np.append(performance_by_dataset,Project)
    MyModelName='DEGNextModel_'+Project+'.pth'
    model = MyNetwork()
    model.load_state_dict(torch.load(MyModelName))
    model.Lin5=nn.Linear(8,1)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    fine_tune_dataset = CancerDataset_global_pooled_multi_datasets_all.MyDataset(data_type=Project,case='bio-fine-tune',transform=DatasetTransform_global_pooled_all.Compose([DatasetTransform_global_pooled_all.ArrayToTensor()]))
    batch_size = 64
    fine_tune_loader = DataLoader(dataset=fine_tune_dataset, batch_size=batch_size, shuffle=True)
    samples=fine_tune_dataset.len
    accuracy1,recall1, precision1, fmeasure1, mcc1=fine_tune_train(model,samples)
    Fine_tune_results='DEGNext Fine tune results for '+Project+': '+str(accuracy1)+'  '+str(recall1)+'    '+str(precision1)+'   '+str(fmeasure1)+'  ' +str(mcc1)
    print(Fine_tune_results)
    print("DEGNext Fine tuning for ",Project," ended....")
    MyModelName='DEGNextModel_'+Project+'.pth'
    torch.save(model.state_dict(), MyModelName)

    print("Testing started............")
    test_dataset = CancerDataset_global_pooled_multi_datasets_all.MyDataset(data_type=Project,case='bio-test',transform=DatasetTransform_global_pooled_all.Compose([DatasetTransform_global_pooled_all.ArrayToTensor()]))
    batch_size = 256
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    accuracy2,recall2,precision2,fmeasure2,mcc2=test(model,Project)
    Test_results='DEGNext Bio test results for '+Project+': '+str(accuracy2)+'  '+str(recall2)+'    '+str(precision2)+'   '+str(fmeasure2)+'  ' +str(mcc2)
    print(Test_results)
    performance_by_dataset=np.append(performance_by_dataset,Test_results)

    dataset="Dataset: "+Project+" "
    auc_file=np.append(auc_file,dataset)

    fpr, tpr, thresh = metrics.roc_curve(y_test, pred_0)
    auc = metrics.roc_auc_score(y_test, pred_0)
    plt.plot(fpr,tpr,label="DEGNext, auc="+str(round(auc, 2)))
    auc_ML=" AUC for DEGNext: "+str(round(auc, 2))
    auc_file=np.append(auc_file,auc_ML)

       ################################################
    plt.legend(loc=0)
    plt.savefig(Project+'_ROC_Exp_2.png')
    plt.clf()

    File='Performance_Exp_2_'+Project+'.txt'
    np.savetxt(File, performance_by_dataset, fmt='%s')
    File='AUC_per_dataset_Exp_2_'+Project+'.txt'
    np.savetxt(File, auc_file, fmt='%s')








