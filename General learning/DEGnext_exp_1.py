#DEGnext 
#Copyright Tulika Kakati 2020 
#Distributed under GPL v3 open source license
#This code may be used to train and test the model on T1 and T2 data for all the cancer datatsets and save the model for each cancer dataset.
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

epoch=50
learning_rate=1e-4
np.random.seed(0)
torch.manual_seed(0)
loss_func = nn.CrossEntropyLoss()
performance_by_dataset=[]
auc_file=[]

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
        m = nn.Softmax(dim=1)
        output = m(x)
        return output


def calculate_accuracy_from_cm(Y_t,y_hat_class):
    cm_accuracy=accuracy_score(Y_t, y_hat_class,normalize=True)
    cm_recall=recall_score(Y_t, y_hat_class, average='weighted')
    precision=precision_score(Y_t, y_hat_class, average='weighted')
    cm_f_measure=f1_score(Y_t, y_hat_class, average='weighted')
    cm_mcc=matthews_corrcoef(Y_t,y_hat_class)
    return cm_accuracy*100, cm_recall*100, precision*100, cm_f_measure,cm_mcc

def train(model,samples):
    training_loss = []
    training_accuracy = []

    for i in range(epoch):
        training_accuracy = []
        training_recall = []
        training_precision = []
        training_fmeasure = []
        training_mcc=[]
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
            loss = loss_func(y_hat,varr)
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            optimizer.zero_grad()
            _,predicted=torch.max(y_hat,1)
            y_hat_class = predicted
            cm_accuracy, cm_recall, precision, f_measure,mcc =calculate_accuracy_from_cm(y_img.numpy(),y_hat_class)
            training_accuracy.append(cm_accuracy)
            training_recall.append(cm_recall)
            training_precision.append(precision)
            training_fmeasure.append(f_measure)
            training_loss.append(loss)
            training_mcc.append(mcc)
        cum_accuracy=sum(training_accuracy)/len(training_accuracy)
        cum_recall=sum(training_recall)/len(training_recall)
        cum_precision=sum(training_precision)/len(training_precision)
        cum_fmeasure=sum(training_fmeasure)/len(training_fmeasure)
        cum_loss = sum(training_loss)/len(training_loss)
        cum_mcc=sum(training_mcc)/len(training_mcc)
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
        id=torch.arange(0,y_hat.shape[0])
        varr=y_img.flatten().long()
        true_label = torch.zeros(y_hat.shape[0],3)
        true_label[id,varr]=1
        loss = loss_func(y_hat,varr)
        test_loss.append(loss.item())
        _,predicted=torch.max(y_hat,1)
        y_hat_class = predicted
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

model = MyNetwork()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

CancerProjects=["TCGA-CHOL","TCGA-COAD","TCGA-ESCA","TCGA-HNSC","TCGA-KICH"]

for i in range(len(CancerProjects)):
    Project=CancerProjects[0]
    performance_by_dataset=[]
    auc_file=[]
    print("Training for ",Project," started....")
    performance_by_dataset=np.append(performance_by_dataset,Project)

    train_dataset = CancerDataset_global_pooled_multi_datasets_all.MyDataset(data_type=Project,case='train',transform=DatasetTransform_global_pooled_all.Compose([DatasetTransform_global_pooled_all.ArrayToTensor()]))
    batch_size = 256
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    samples=train_dataset.len
    accuracy0,recall0, precision0, fmeasure0, mcc0=train(model,samples)
    Train_results='Train results for '+Project+': '+str(accuracy0)+'  '+str(recall0)+'    '+str(precision0)+'   '+str(fmeasure0)+'  ' +str(mcc0)
    performance_by_dataset=np.append(performance_by_dataset,Train_results)
    print(Train_results)
    print("Training for ",Project," ended....")

    MyModelName='DEGNextModel_'+Project+'.pth'
    torch.save(model.state_dict(), MyModelName)



    print("Testing started............")
    MyModelName='DEGNextModel_'+Project+'.pth'
    my_model = MyNetwork()
    my_model.load_state_dict(torch.load(MyModelName))
    my_model.eval()
    print("Model loaded.....")

    test_dataset = CancerDataset_global_pooled_multi_datasets_all.MyDataset(data_type=Project,case='non-bio-test',transform=DatasetTransform_global_pooled_all.Compose([DatasetTransform_global_pooled_all.ArrayToTensor()]))
    batch_size = 256
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    accuracy2,recall2,precision2,fmeasure2,mcc2=test(model,Project)
    Test_results='DEGNext Non-bio test results for '+Project+': '+str(accuracy2)+'  '+str(recall2)+'    '+str(precision2)+'   '+str(fmeasure2)+'  ' +str(mcc2)
    print(Test_results)
    performance_by_dataset=np.append(performance_by_dataset,Test_results)

    
    dataset="Dataset: "+Project+" "
    auc_file=np.append(auc_file,dataset)
    File = Project+'_non_bio_test_label.txt'
    xy_label= np.loadtxt(File,delimiter=',', dtype=np.float32)
    y_test = label_binarize(xy_label, classes=[0, 1, 2])
    n_classes = y_test.shape[1]

    File = 'Test_predicted_probs_'+str(Project)+'.txt'
    y_pred = np.loadtxt(File,delimiter=',', dtype=np.float32)
    prob_predict3D=y_pred.reshape((-1,3))
    pred_0=prob_predict3D

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_0[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred_0.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    auc1 = roc_auc["micro"]
    

    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',lw=lw, label='DEGNext, auc='+str(round(auc1, 2)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    auc_ML="AUC for DEGNext: "+str(round(auc1, 2))
    auc_file=np.append(auc_file,auc_ML)
    plt.legend(loc=0)
    plt.savefig(Project+'_ROC_comparison_Exp_1.png')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.clf()


    File='Performance_Exp_1_'+Project+'.txt'
    np.savetxt(File, performance_by_dataset, fmt='%s')
    File='AUC_per_dataset_Exp_1_'+Project+'.txt'
    np.savetxt(File, auc_file, fmt='%s')








