# DEGnext
Tulika Kakati
DEC 05, 2020
# Citation
Please use the source code ethically.
# Prerequisites for testing platform
pytorch (1.3 and above), sklearn (0.21.3 and above)
# Introduction
DEGnext is a CNN model, to predict upregulating (UR) and downregulating (DR) genes from gene expression data of 17 cancer datasets obtained from The Cancer Genome Atlas (TCGA) database. The DEGnext uses biologically validated data along with logarithmic fold change (logFC) values to classify differentially expressed genes (DEGs) as UR and DR genes. We applied transfer learning to our model to leverage the knowledge of trained feature maps to untrained cancer datasets. 
Here, we show the effectiveness of our model on 5 cancer datasets, namely "TCGA-CHOL","TCGA-COAD","TCGA-ESCA","TCGA-HNSC","TCGA-KICH". 
# Experiment 1 (General Learning):} 
In the first experiment,   we used all 5 cancer datasets to train (T1 data),   fine-tune (F1 data) and test the corresponding T3 bio-test data from each dataset. Since the T1 data has three labels `0',   `1',   and `2',   this training is for a three-class problem. DEGnext uses CrossEntropyLoss() as a loss function and optim.Adam() as an optimizer to compute the cross entropy loss between the output for a given input and updates the parameters based on the gradients. For predicted classes 0,   1,   2,   the input gene is classified as DR,   UR or neutral gene. 

For the second level of training,   we fine-tune on F1 data. Since F1 data have `0' and `1' labels,   the second level of training is a two-class problem. Here,   we used the BCEWithLogitsLoss() loss function to fine tune the model. After fine-tuning, DEGnext model is then tested using T3 bio test data. The second level of training incorporates both prior disease-related biological knowledge and log2FC estimates (sample variance) of the data to the CNN model,   which enables capture of non-linear gene expression patterns and enhances prediction performance of the model in determining UR and DR genes. The major advantage of our CNN model is that it allows performing very efficient transfer learning by reusing the feature-map signatures learned from the trained model.\\
# Experiment 2 (Transfer learning): 
For the second experiment,   we used 3 training datasets: "TCGA-CHOL","TCGA-COAD","TCGA-ESCA" and 2 test datasets: "TCGA-HNSC","TCGA-KICH". We trained on  T1 data for all 3 training  datasets one after another. For training, since T1 data has three labels, `0', `1', and `2', CrossEntropyLoss() as a loss function and optim.Adam() as an optimizer, with a batch size of 256. For fine tune, all we needed to do is to customize and modify the output layer L5 and remove the final softmax layer to classifying the DEGs as `0' or `1'. We used the BCEWithLogitsLoss() loss function to fine tune the model again with the F1 data for all 3 training datasets. For testing, since the data similarity among the training and testing datasets is very high, we do not need to retrain the model.  We used the pretrained model as a feature extractor and leveraged the knowledge acquired from the trained model to predict UR and DR genes from bio-test data of both the test datasets.
