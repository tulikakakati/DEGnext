
#DEGnext 
#Copyright Tulika Kakati, 2021
#Distributed under GPL v3 license
#This code may be used to download and preprocess cancer RNA-seq datasets from TCGA portal. 

############################################
CancerProject=c("TCGA-BRCA","TCGA-KIRC","TCGA-UCEC","TCGA-LUAD","TCGA-THCA","TCGA-LUSC","TCGA-PRAD","TCGA-BLCA","TCGA-STAD","TCGA-LIHC","TCGA-READ","TCGA-ESCA","TCGA-THCA","TCGA-HNSC","TCGA-COAD")
library("TCGAbiolinks")
library("SummarizedExperiment")
memory.limit(150000000000)
		for(i in 1:length(CancerProject)){
		Project=CancerProject[i]				  
		query <- GDCquery(project = Project,
						  data.category = "Transcriptome Profiling",
						  data.type = "Gene Expression Quantification", 
						  workflow.type = "HTSeq - Counts",
						  sample.type = c("Primary Tumor","Solid Tissue Normal"),
						  legacy = FALSE)

		GDCdownload(query,method = "api",files.per.chunk = 150)

		cat("data preparation starts......\n")
		# Prepare expression matrix with geneID in the rows and samples (barcode) in the columns
		# rsem.genes.results as values
		BRCARnaseqSE <- GDCprepare(query,
								save = FALSE, 
								summarizedExperiment = TRUE)

		cat("data preparation ends......\n")
		BRCAMatrix <- assay(BRCARnaseqSE,"HTSeq - Counts") # or BRCAMatrix <- assay(BRCARnaseqSE,"raw_count")
		dim(BRCAMatrix) #

		
		group1 <- TCGAquery_SampleTypes(colnames(BRCAMatrix), typesample = c("NT"))
		group2 <- TCGAquery_SampleTypes(colnames(BRCAMatrix), typesample = c("TP"))
		length(group1)
		length(group2)

		# For gene expression if you need to see a boxplot correlation and AAIC plot to define outliers you can run
		cat("data preprocessing starts.......\n")
		BRCARnaseq_CorOutliers <- TCGAanalyze_Preprocessing(BRCARnaseqSE,
												  cor.cut = 0.6, 
												  datatype = "HTSeq - Counts",
												  filename = "GBM_IlluminaHiSeq_RNASeqV2.png")

		dim(BRCARnaseq_CorOutliers) 
		cat("data preprocessing ends......\n")
		
		library("biomaRt")
		mart <- useMart("ENSEMBL_MART_ENSEMBL")#use this or the below command as and when required
		#ensembl = useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl", mirror = "useast")
		mart <- useDataset("hsapiens_gene_ensembl", mart)
		pre_ensembles=rownames(BRCARnaseq_CorOutliers)
		dftmp=getBM(filters="ensembl_gene_id",attributes=c("ensembl_gene_id","hgnc_symbol"),values=pre_ensembles,mart=mart)
		valid_ensembles=dftmp$ensembl_gene_id
		valid_ensembles=unique(valid_ensembles)
		length(valid_ensembles)# 


		ensembl_ind=match(valid_ensembles, pre_ensembles)
		genes=dftmp$hgnc_symbol
		genes=genes[ensembl_ind]
		genes=genes[!is.na(genes)]
		length(genes)
		valid_genes=as.character()
		for(i in j:length(genes)){
			d=nchar(genes[j])
			if(d>0){
				valid_genes=c(valid_genes,genes[j])
			}
		}
		length(valid_genes)#

		genes_ind=match(valid_genes, genes)
		length(genes_ind)#
		BRCARnaseq_CorOutliers=BRCARnaseq_CorOutliers[genes_ind,]
		rownames(BRCARnaseq_CorOutliers)=valid_genes
		dim(BRCARnaseq_CorOutliers)
		


		cat("ensembl names converted to gene names.....\n")


		# Downstream analysis using gene expression data  
		# TCGA samples from IlluminaHiSeq_RNASeqV2 with type rsem.genes.results
		# save(dataBRCA, geneInfo , file = "dataGeneExpression.rda")
		library(TCGAbiolinks)
		cat("Data normalization starts.....\n")
		# normalization of genes
		dataNorm <- TCGAanalyze_Normalization(tabDF = BRCARnaseq_CorOutliers,geneInfo=geneInfo)
		dim(dataNorm)
		

		cat("Data normalization ends.....\n")
		cat("Data filtering starts.....\n")
		# quantile filter of genes
		dataFilt <- TCGAanalyze_Filtering(tabDF = dataNorm,
										  method = "quantile", 
										  qnt.cut =  0.25)

		dim(dataFilt)
		


		cat("Data filtering ends.....\n")



		# selection of normal samples "NT"
		samplesNT <- TCGAquery_SampleTypes(barcode = colnames(dataFilt),
										   typesample = c("NT"))
										   
		# selection of tumor samples "TP"
		samplesTP <- TCGAquery_SampleTypes(barcode = colnames(dataFilt), 
										   typesample = c("TP"))
		normal_samples=dataFilt[,samplesNT]
		tumor_samples=dataFilt[,samplesTP]
		File=sprintf("%s_normal_samples.csv",Project)
		write.table(normal_samples,File,sep=",",col.names=NA)
		File=sprintf("%s_tumor_samples.csv",Project)
		write.table(tumor_samples,File,sep=",",col.names=NA)
										   
										   
}
