# Biological read across using deep embedding of gene expression profiles

## L1000
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5990023/

Having validated L1000, we set out to expand on the CMap pilot dataset in several dimensions. First, we increased the small molecule perturbations from 164 drugs to 19,811 small molecule drugs, tool compounds and screening library compounds including those with clinical utility, known mechanism of action, or nomination from the NIH Molecular Libraries Program. Each compound was profiled in triplicate, either at 6 or 24 hours following treatment.

Second, we expanded in the dimension of genetic perturbation by knocking down and overexpressing 5,075 genes selected on the basis of their association with human disease or membership in biological pathways. Each genetic perturbation was profiled in triplicate, 96 hours after infection. For overexpression studies, a single cDNA clone was used, whereas three distinct shRNAs targeting each gene were profiled.

Third, we expanded in the dimension of cell lines. Well-annotated genetic and small molecule perturbagens were profiled in a core set of 9 cell lines, yielding a reference dataset we refer to as Touchstone v1. Uncharacterized small molecules without known mechanism of action (MOA) were profiled variably across 3 to 77 cell lines, yielding a dataset we refer as Discovery v1 (Table S4).

In total, we generated 1,319,138 L1000 profiles from 42,080 perturbagens (19,811 small molecule compounds, 18,493 shRNAs, 3,462 cDNAs, and 314 biologics), corresponding to 25,200 biological entities (19,811 compounds, shRNA and/or cDNA against 5,075 genes, and 314 biologics) for a total of 473,647 signatures (consolidating replicates), representing over a 1,000-fold increase over the CMap pilot dataset.

Phase I and II level 4 (z-scores) data for a total of 1,674,074 samples corresponding to 44,784 perturbagens were retrieved from GEO (accession numbers GSE92742 and GSE70138).

## Download data

https://clue.io/releases/data-dashboard has wget code for data pull

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138

https://clue.io/data/CMap2020#LINCS2020

> Level 4 (Z-SCORES) - signatures with differentially expressed genes computed by robust z-scores for each profile relative to control (PC relative to plate population as control; VC relative to vehicle control). 

(Level 4 does not have collapsed replicates)

Can make python script with
```
import urllib.request

data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
                           filename=os.path.join(data_folder, 'train-images-idx3-ubyte.gz'))
```

## Data dict
For level 4 data `GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx`

Each row in gctx `data_df` represents a gene
the row id matches with `rid` in `GSE92742_Broad_LINCS_gene_info.txt`

Each column in gctx `data_df` represents a sample
The col id matches with `cid` in `GSE92742_Broad_LINCS_gene_info.txt`

## Load data (.gctx files)

* GSE92742 and GSE70138 loaded in as GCToo object 

* extract landmark (LM) probes from imputed probes

* Removal of 106,499 control samples (as indicated by `sm_pert_type` field)

* Add annotations, see: https://github.com/cmap/cmapPy/blob/master/tutorials/cmapPy_pandasGEXpress_tutorial.ipynb 

* Concat using cmapPy

> After removal of 106,499 control samples (as indicated by the `sm_pert_type` field), 1,567,575 samples corresponding to 44,693 perturbagens remained (left in genetic manipulations). Our analysis only included the 978 L1000 landmark genes and did not use imputations.

https://clue.io/connectopedia/perturbagen_types_and_controls
```
['ctl_vehicle', 'trt_cp', 'ctl_untrt', 'trt_sh.cgs',
       'ctl_vehicle.cns', 'ctl_vector.cns', 'ctl_untrt.cns', 'trt_sh.css',
       'trt_lig', 'ctl_vector', 'trt_sh', 'trt_oe', 'trt_oe.mut']
```

Just controls
```
['ctl_vehicle'
'ctl_vector'
'trt_sh.css'
'ctl_vehicle.cns'
'ctl_vector.cns'
'ctl_untrt.cns '
'ctl_untrt']
```

| Perturbagen Type                                                             | pert\_type designation in metadata files |
| ---------------------------------------------------------------------------- | ---------------------------------------- |
| Compound                                                                     | trt\_cp                                  |
| Peptides and other biological agents (e.g. cytokine)                         | trt\_lig                                 |
| shRNA for loss of function (LoF) of gene                                     | trt\_sh                                  |
| Consensus signature from shRNAs targeting the same gene                      | trt\_sh.cgs                              |
| cDNA for overexpression of wild-type gene                                    | trt\_oe                                  |
| cDNA for overexpression of mutated gene                                      | trt\_oe.mut                              |
| CRISPR for LLoF                                                              | trt\_xpr                                 |
| Controls - vehicle for compound treatment (e.g DMSO)                         | ctl\_vehicle                             |
| Controls - vector for genetic perturbation (e.g empty vector, GFP)           | ctl\_vector                              |
| Controls - consensus signature from shRNAs that share a common seed sequence | trt\_sh.css                              |
| Controls - consensus signature of vehicles                                   | ctl\_vehicle.cns                         |
| Controls - consensus signature of vectors                                    | ctl\_vector.cns                          |
| Controls - consensus signature of many untreated wells                       | ctl\_untrt.cns                           |
| Controls - Untreated cells                                                   | ctl\_untrt                               |

### Controls

https://clue.io/connectopedia/perturbagen_types_and_controls

> In the current CMap data processing workflow that computes differential expression for each perturbation, we use a population control, which represents all other perturbagens on the same physical plate. We have found that use of the population control results in robust differential expression signatures, in that the value for each gene indicates how much it was affected by a specific perturbagen relative to a diverse collection of other perturbagens on the same plate.

> We also include control perturbations in our experimental designs for compound and genetic perturbations, and these controls are added to most plates. Vehicle control (designated `ctl_vehicle`/`ctl_untrt`) refers to the solvent used to administer compound treatments, which is usually DMSO. Vector control, or `ctl_vector`, is a negative control that refers to genetic perturbagens that either do not contain a gene-specific sequence, or whose gene-specific sequence targets a gene not expressed in the human genome (such as GFP or RFP). In all cases, negative controls are expected to be largely inert and therefore should not cause notable gene expression changes, and use of these controls helps us to monitor the technical fidelity of the experiment.

## Data pre-processing for training

Standardization is performed per gene by subtracting the mean and dividing by the standard deviation
Means and variances are estimated over the entire training set.

Gaussian noise with a standard deviation of 0.3 to the input.

## Network architecture

> The output embeddings are used to predict the class (perturbagen identity) of the
input by softmax where logits are cosine similarities between the
profile embeddings and learned class embeddings, scaled by a learned
constant. The prediction cross-entropy loss is used to train the
network.

> The network used in our experiments
includes 64 hidden layers, but when we compared networks
with different depths and approximately the same number of
parameters, we found very little, if any, improvement beyond
four hidden layers.

### Dense vs Convolutional layers

Dense Connections, or Fully Connected Connections, are a type of layer in a deep neural network that use a linear operation where every input is connected to every output by a weight. This means there are  parameters, which can lead to a lot of parameters for a sizeable network.

![dense](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-27_at_6.31.32_PM_xBfVMWZ.png)

Dense layer: A linear operation in which every input is connected to every output by a weight (so there are n_inputs * n_outputs weights - which can be a lot!). Generally followed by a non-linear activation function

Convolutional layer: A linear operation using a subset of the weights of a dense layer. Nearby inputs are connected to nearby outputs (specifically - a convolution 859 ). The weights for the convolutions at each location are shared. Due to the weight sharing, and the use of a subset of the weights of a dense layer, there’s far less weights than in a dense layer. Generally followed by a non-linear activation function

Tabular data doesn't have local structure (eg. images where nearby pixel values tend to be highly correlated). Tabular data do not benefit from convolution because there is typically no spatial structure there.

### DenseNET

DenseNET: 
https://d2l.ai/chapter_convolutional-modern/densenet.html
https://amaarora.github.io/2020/08/02/densenets.html

https://arthurdouillard.com/post/densenet/

Non-convolutional linear densenet (Keras)
https://github.com/DowellChan/DenseNetRegression/blob/main/OptimalDenseNetRegression.py

### Growth rate
https://d2l.ai/chapter_convolutional-modern/densenet.html
> The number of convolution block channels controls the growth in the number of output channels relative to the number of input channels. This is also referred to as the growth rate (k)

### Self normalising network (selu activation)

> SGD and dropout perturb these kinds of normalisation (and they can be tricky to code), leading to high variance in training error. CNNs and RNNs get around this by sharing weights (though RNNs are still subject to exploding/vanishing gradients). The effect gets worse with depth, so deep vanilla networks tend to suck.
https://medium.com/@damoncivin/self-normalising-neural-networks-snn-2a972c1d421

SNNs are already implemented in PyTorch and Keras

### Memory efficient densenet

https://github.com/joeyearsley/efficient_densenet_tensorflow

### ArcFace loss

https://github.com/peteryuX/arcface-tf2
https://github.com/4uiiurz1/keras-arcface

Good guide to training ArcFace: https://medium.datadriveninvestor.com/a-hackers-guide-to-efficiently-train-deep-learning-models-b2cccbd1bc0a

#### Trainable scale parameter

Trainable layer parameter
https://keras.io/guides/making_new_layers_and_models_via_subclassing/

#### Linear increasing of margin parameter with training steps

https://stackoverflow.com/questions/63630875/change-keras-model-variable

### Training 

#### Data loading for training

Training data cannot fit in the GPU memory which is needed at the start of each fit epoch.

Keras recommends use of `tf.data.Dataset` objects to load data in a multi-device or distributed workflows. With a python generator, GPU useage was very poor for large batch sizes and peaky (data starvation?)
https://www.tensorflow.org/tutorials/load_data/numpy

Warning with input shape
https://stackoverflow.com/questions/67638345/tensorflow-dataset-with-multiple-inputs-and-target/67639990#67639990

### Recovering data from an Experiment run 

Get data from previous experiments: https://stackoverflow.com/questions/66020144/how-can-one-download-the-outputs-of-historical-azure-ml-experiment-runs-via-the

Could use in conjunction with keras checkpointing: https://keras.io/guides/distributed_training/
Manually download run files, then provide directory to commandline option for training script

## Evaluation

> We trained a deep neural network to embed L1000 expression profiles in a metric space where similarity
calculated as the dot product between embeddings, clusters expression profiles by perturbagen

> Cosine similarity only cares about angle difference, while dot product cares about angle and magnitude. If you normalize your data to have the same magnitude, the two are indistinguishable

> Developed a method for embedding L1000
expression profiles into the unit hypersphere {x ∈ Rn |∥x∥2 =
1} using a deep neural network. In this space, a similarity
between −1 and 1 is obtained by the dot product of two
embedded profiles.

For each profile in the held-out subset, compute its similarity to all other profiles from 
-the same perturbagen (positives) 
-profiles from other perturbagens(negatives) 
that are also part of the held-out subset.

Think: dot product of generated embeddings for
-same peturbagen
-replicate
inside test set

Recall is how many times ranked the positive in that quantile (divided into equal-sized adjacent subgroups)
eg. 0.2 quantile means top 20%

Compare against naive model (cosine similarity)
Compare against connectivity score (https://clue.io/connectopedia/cmap_algorithms)

Hopefully some of the peturbagens overlap with Biospyder chemicals!
CMap chems are mostly small molecules

"SigCom LINCS data and metadata search engine for a million gene expression signatures"
Uses more performant similarity with mann whitney U test

### Weighted ROC
Weighted Area Under the Receiver Operating Characteristic Curve and Its Application to Gene Selection https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4129959/

## Saving a Keras model

Need to be careful with custom layers
https://linuxtut.com/en/b2154b661e7baf56031e/

Azure ML: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-tensorflow#register-or-download-a-model
https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/tensorflow/train-hyperparameter-tune-deploy-with-tensorflow/tf_mnist.py

## Ideas for extension

### Other network architectures

Kaggle MoA prediction, https://www.kaggle.com/c/lish-moa
Use of TabNet and 1D cov: https://www.kaggle.com/c/lish-moa/discussion/202256

#### 1D-CNN
Soft-Ordering 1-dimensional CNN. The idea of introducing order before the convolution layers
https://github.com/broadinstitute/lincs-profiling-complementarity/blob/master/2.MOA-prediction/3.moa_prediction_models/pytorch_model_helpers/pytorch_utils.py
Tensorflow implementation: https://www.kaggle.com/code/sishihara/1dcnn-for-tabular-from-moa-2nd-place/notebook

PCA feature are used for all models with different n_components (50 genes + 15 cells) for 1D-CNN
Dummy variable (cp_time, cp_dose) are removed

#### TabNet
Tensorflow implementations:
https://github.com/jeyabbalas/tabnet (best with example)
https://github.com/ostamand/tensorflow-tabnet
https://github.com/titu1994/tf-TabNet
https://github.com/google-research/google-research/tree/master/tabnet

#### Others

https://www.sciencedirect.com/science/article/pii/S1043661822001700
The DNN parameters were adopted as follows: three hidden layers and fully connected structures, the corresponding hidden layer nodes are 978, 512, and 256. Activation function=‘relu’, dropout=0.1, learning rate=0.001, iteration=2000. 
Training set labels: Drugs with the same MOA annotation (refer to MedChemExpress drug library) were used as positive training samples, and 103 MOAs were collected. The negative samples were pre-defined 6220 compounds that without MOA annotations (refer to both MedChemExpress drug library and Drug Repurposing Hub

https://github.com/sgfin/molecule_ge_coordinated_embeddings
Gene expressions are embedded using self-normalizing neural networks, which are feedforward
neural networks that employ the SeLU activation function. The depth and shape of the
hidden layers was left, in principle, as a hyperparameter, though in practice we generally
found two hidden units of size 1024 and 512 between the input and embedding layers to
give adequate performance. 

### Misc ideas
https://hav4ik.github.io/articles/deep-metric-learning-survey

Class imbalance was resolved by 1st place (https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176037) using a weighted cross-entropy on top of ArcFace. 5th place (https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176151) proposed another approach, using Focal Loss with Label Smoothing on top of ArcFace.
3rd place winners (recognition) used Sub-Center ArcFace with Dynamic Margin. https://hav4ik.github.io/articles/deep-metric-learning-survey#subcenter-arcface

Kaggle: Shopee Price Match Guarantee (2021)
Given an image of some product and its description, the task is to find similar products in the test set (not seen during training) by their images and text descriptions.
https://www.kaggle.com/code/nadare/adacos-curricularface-for-tf/notebook

- We removed samples related to cell lines in which less than 5000 profiles were measured "AI for the repurposing of approved or investigational drugs against COVID-19"
- Include the extra L1000 beta release samples
- Transfer learning by pre-training on genetic perturbagens rather than all at once?
- Pre-processing data, Label Smoothing (Kaggle MoA https://www.kaggle.com/c/lish-moa/discussion/202256)
- Hyperbolic embeddings
    - https://medium.com/@nathan_jf/treerep-and-hyperbolic-embeddings-41312c98b264
    - https://github.com/nalexai/Hyperlib
- Include chemical structure with mol2vec https://mol2vec.readthedocs.io/en/latest/. Concatenate embeddings for each of the sequence with other tabular features.
- Better than averaging over replicates (like paper suggests)? (Use moderated z-score https://clue.io/connectopedia/replicate_collapse)
- Dose, timepoint?

> *[09:33] Scott, Andrew*
Liddell, Mark Biological Read-Across using gene expression profiles was an approach that the Human Biology team (lead by Paul van der Logt) was interested in for Natural Extracts.  Maja linked with this a few years ago from a SEAC RA pov.  Are Natural Extracts in scope of the datasets you're working with?

Production: model can transform RNA-seq to L1000 (do for biospyder)
"SigCom LINCS: data and metadata search engine for a million gene expression signatures"
> One generates RNA-seq profiles from L1000 profiles, and the other generates L1000 profiles from RNA-seq profiles. The first generator takes L1000 profiles as input and outputs RNA-seq profiles

## Presentation

algorithm that uses artificial intelligence (deep learning) to identify groups of small molecules that have similar functional impact on human biology.

The goal of Metric Learning is to learn a representation function that maps objects into an embedded space. The distance in the embedded space should preserve the objects’ similarity — similar objects get close and dissimilar objects get far away. 

https://www.sciencedirect.com/science/article/pii/S1043661822001700
> In the past decade, large scale gene expression data has been accumulated due to the emergence of many high-throughput technologies. Particularly, the Connectivity Map [1] (CMap), including its upgraded version L1000 database[2], provides one of the most important platforms for in silico screening by connecting genes, drugs and diseases via gene expression profile datasets. When CMap data is applied in drug repurposing, measuring the similarity between different signatures is a common way to identify shared MOAs of otherwise dissimilar drugs (drugs that belong to different categories or structures). This principle is called "guilt by association"[3], [4], and this strategy could aid to identify alternative targets of existing drugs and uncover potential off-target effects that can be investigated for clinical applications

> The principle of "guilt by association" assumes that drugs with similar gene expression signatures may share the same targets, MOAs, adverse effects and even indications. Thus comparing pairwise similarity between signatures are commonly used to explore the unknown properties of queried compounds. 
However pairwise similarity comparisons are often easily disturbed by data noise.

The prediction classifier is not aiming for “classification” but for ranking the probability of each signature. Usually only the highest ranked (such as top10 or top50) prediction molecules may be worth further bioassay validation.

- Goal of the method is to rank perturbagens in silico based on functional similarity to a specific query perturbagen

embeddings of pharmacologically similar perturbagens
would be closer in the trained embedding space than those of
pharmacologically dissimilar perturbagens.

- Work up from toy cosine similarity example

- L1000 dataset

- Quite some challenges: batch effects, dose, cell line, time point

- Introduce embeddings
semantically or functionally related items to have closer representations (and conversely semantically or functionally unrelated items to have more distant representations) in the embedding space

How can I build a numerical representation of a gene expression profile that can efficiently embed its characteristics and be used in similarity tasks?

Cross-entropy only learns how to map an image to a label, without learning the relative distances (or similarities) between the inputs.
When you need an embedding for visual similarity tasks, your network should explicitly learn, at training time, how to compare and rank items between each other. The task that learns efficient embeddings that compare and rank inputs between each other is called metric learning.

- ArcFace
    - ArcFace loss results in feature extraction with better intra-class compactness and inter-class discrepancy
    - main goal is to maximize face class separability by learning highly discriminative features for face recognition
    - Visualization of Embeddings trained with and without ArcFace layer (https://www.kaggle.com/code/dongdori/arcface-versus-cross-entropy-better-embeddings/notebook)
    - Good description of ArcFace: https://learnopencv.com/face-recognition-with-arcface/

the metric was capable of capturing pharmacological similarity even among compound pairs with low structural similarity

"But gene expression in cell lines are very different for each chemical"
-model performance with and without including cell line
-L1000 data has mostly __ cell lines per chemical (overlap with ours?) [see Harnessing the biological complexity of Big Data from LINCS gene expression signatures]
-inclusion in tabnet change of performance (also look at tabnet expandability)

## Misc
"SigCom LINCS data and metadata search engine for a million gene expression signatures"
UMAP plots that visualize only the chemical perturbations (Figure 2B)
These plots show that some perturbations are cell type-specific and some cell type agnostic.

## Azure ML

### Ways of working

Can use local machine to develop. Use custom environments with https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-terminal#add-new-kernels
Interactive logging session, code will run in the local Azure ML Compute instance. 
```
from azureml.core import Experiment 

# create an experiment variable 
experiment = Experiment(workspace = ws, name = "my-experiment") 

# start the experiment 
run = experiment.start_logging() 

# experiment code goes here 

# end the experiment 
run.complete()
```
```
import azureml

from azureml.core import Experiment
from azureml.core import Workspace

from azureml.core import ScriptRunConfig

ws = Workspace.from_config()

args = ['--msg', "lol"]

src = ScriptRunConfig(source_directory='.',
                      script='test.py',
                      arguments=args)

run = Experiment(workspace=ws, name='msg').submit(src)
run.wait_for_completion(show_output=True)
```

Then bring in compute cluster with `compute_target=`, use custom envs with `environment=`