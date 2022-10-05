# Biological read across using deep embedding of gene expression profiles

This repo contains code to replicate the deep learning model described in *Drug Repurposing Using Deep Embeddings of Gene Expression Profiles* (https://doi.org/10.1021/acs.molpharmaceut.8b00284) using the Azure ML platform.

## Introduction

A deep learning model measures functional similarities between compounds based on gene expression data for each compound. The model receives an unlabeled expression profile for a query perturbagen including transcription counts of a plurality of genes in a cell affected the query perturbagen. The model extracts an embedding of the expression profile. 

Using the embedding of the query perturbagen and embeddings of known perturbagens, the model determines a set of similarity scores, each indicating a likelihood that a known perturbagen has a similar effect on gene expression as the query perturbagen. The likelihood, additionally, provides a prediction that the known perturbagen and query perturbagen share pharmacological similarities. 

The similarity scores are ranked and, from the ranked set, at least one candidate perturbagen is determined to be pharmacologically similar to the query perturbagen. The model may further be applied to determine similarities in structure and biological protein targets between perturbagens.

## Use

```
pip install -r requirements.txt
```
(requirements.txt has -e in it for editable development)

Then run this script from the compute instance
```
python src/deep_embeddings/azureml/aml_create_and_trigger_pipeline.py
```

## References

Azure ML pipeline example: https://medium.com/datamindedbe/ml-pipelines-in-azure-machine-learning-studio-the-right-way-26b1f79db3f8

Relevant Azure ML docs for creating pipelines: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines

Developing efficiently on Azure ML: https://azure.github.io/azureml-cheatsheets/docs/cheatsheets/python/v1/ci-dev/