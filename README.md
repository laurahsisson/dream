# Overview
The goal of my approach was to pre-train and then transfer-learn a large graph neural network model, with a novel molecule featurization technique, multiple MPNN layers, and a SetTransformer readout function. This model would be pre-trained on the 266k aroma-chemical pairs from “Olfactory Label Prediction on Aroma-Chemical Pairs” and then fine-tuned on the DREAM mixture similarity data.

During pre-training, the GNN model would learn to embed molecules and predict the discrete labels for the aroma-chemical pairs, and as in “Expansive linguistic representations to predict interpretable odor mixture discriminability”, the notes and/or embeddings would be used as inputs to the similarity prediction model. Because of the limited time/compute, the hyperparameter tuning trials for pre-training and fine-tuning were conducted separately.

## Repository Contents
This repository contains:
* `./`: The utilties and model components.
* `Colab/`: The Google Colab notebooks used to tokenize, pre-train, fine-tune and submit the models.
* `Notebooks/`: Various notebooks used to generate and explore the dataset, as well as test the models.
* `Model/`: The Pytorch checkpoints for the pre-trained aroma-chemical pair prediction model, as well as the molecule tokenizing dictionary and the pre-training configuration/results.

## Architecture
The GNN model used to embed graphs consisted of a molecule featurizer, a number of MPNN layers, and then a readout scheme. I used MPNN as the message passing function because it performed better than the GIN message passing function in my previous work. In the previous work, we used a single MPNN layer with 3 message passing steps, followed by a Set2Set readout function. Because of “Transfer learning with graph neural networks for improved molecular property prediction in the multi-fidelity setting”, I proceeded with a SetTransformer readout.

### Message Passing Neural Network
Because I had slightly more data, I attempted to use a number of stacked MPNNs. This was inspired by examining the architectures of early computer-vision models (especially AlexNet), which used a number of stacked convolutional layers, decreasing by width but increasing by number of channels in subsequent layers. 
The final architecture for the graph-convolutional layers consisted of two MPNNs:
* Node hidden-dimension of 16, and edge hidden-dimension of 8, with 4 message passing steps.
* Node hidden-dimension of 64, and edge hidden-dimension of 16, with 2 message passing steps.
This is surprisingly similar to CV papers, as the node/edge hidden dimensions are akin to channels, and the message passing step is akin to stride/width of the convolution. In other hyperparameter trials, the edge dim tended to be higher than the node dim across all layers.

I also attempted to add in an edge-update function that updated the edge hidden-state based on the states of the two nodes it connects, but these models did not perform well during hyperparameter tuning.

This may be because I attempted to update the edge-hidden state every message passing step. Perhaps updating the edge states in between/after each message passing layer would have performed better.

The MPNNs used max aggregation, though I experimented with mean and sum aggregation.

### Readout Function
As noted above, I proceeded with SetTransformer as the readout scheme. Although I am not convinced it is better than Set2Set, the paper “Expansive linguistic representations to predict interpretable odor mixture discriminability” suggested it transferred better to different tasks. If I had the implementation time, I would have attempted a hyper-parameter search across readout functions. I did not attempt mean/max pooling as they did not perform well in the recent papers I read.

In the aroma-chemical pair paper, we explored both two stage readouts (molecule-level pooling + blend-level pooling) and single stage readouts (combining every molecule into a single, multi-component graph). In the aroma-chemical pair paper, the single stage readout worked best. During the hyperparameter search for pre-training this held true again, though I believe this may have limited my model’s performance during transfer learning. 

While treating all atoms during readout as if they belong to a single graph works well for the pair task, where the number of atoms is limited, I think this does not hold true during the blend tasks where a larger number of atoms appear. Because the hyperparameter tuning was conducted separately, the two stage models that might have performed suboptimally in the pair task but succeeded in the blend task were filtered out.
Regardless, the model chosen had 5 SetTransformer layers with 16 heads per layer. This is notably shorter and wider than the equivalent Transformer layers used in natural language processing.

### Featurization
To featurize the molecules, I used the Open Graph Benchmark smiles2graph featurizer, which has 9 atom attributes (Atomic Number, Chirality, Degree, Formal Charge, #Hydrogens, #Radical Electrons, Hybridization, Aromaticity, Is In Ring) and 3 edge attributes (Bond Type, Bond Stereo, Conjugation). Because the total number of combinations of features for atoms and bonds is very small (73 and 11 respectively), I experimented with converting the featurized atoms and bonds into tokens and using an embedding dictionary. I think in further research this will prove quite productive, as projecting these features using linear weights does not take into account the radical difference between atoms/bonds that have even a single difference in their attributes.

The hyperparameter tuning indeed showed this, and the best models usually included this tokenization scheme. After tokenization to 1-hot vectors, the 77 atom tokens were mapped to a 64 dimensional space, and the 11 edge tokens were mapped to 128 dimensional space. In the majority of trials, as with the convolutional layers, the edge dictionary had a larger hidden dimension than the atom dictionary.

### Similarity Prediction
For the similarity prediction task, the model took as input the concatenated blend embeddings and blend note-logits. I experimented with concatenating the inputs for each blend, but I found that subtracting the inputs from one blend to the other worked better. This input was passed through a multi-layer perceptron (hidden dim = 1024) with a sigmoid activation function to predict the blend similarity.

## Training
### Pre-Training
As noted above, the hyperparameter search was split between the pre-training and fine-tuning tasks. The 266k labeled pairs were split using an 80/20 random split. Notably, this dataset was not carved. In other words, though no pairs were shared between the train and test sets, there were molecules that appeared across both. This differs from the original aroma-chemical pair paper, but because the DREAM data also shared molecules across the train/leaderboard/test sets, I focused on evaluating the performance on unseen blends (with previously seen molecules).

The training task was a multi-label classification task for the 130 notes. After 128 pre-training trials (Quasi Monte Carlo Sampler with Hyperband Pruning) for 150 epochs, models which achieved an AUROC above 0.70 were evaluated for fine-tuning.

### Fine-Tuning
To evaluate the performance of the fine-tuned models, I used a leave-one-dataset-out cross-validation scheme. In this scheme, each dataset had a fold where it was used as the test set to evaluate the performance of models trained on the other three sets combined. I used the average RMSE across the cross-validation trials. I attempted to use the class-imbalance between datasets as a weighting scheme for the MSE loss, but this was not helpful.

Although I attempted to use a variety of fine-tuning hyperparameter searches, the best models from these searches did not transfer well to the leaderboard set. As a result, I did a fair bit of manual hyperparameter tuning, choosing the best performing model from pre-training as the base model.

## Post Mortem
The results from this extensive training scheme were quite disappointing. Although I came into the competition with a strong dataset advantage and a decent sense of architecture, I was unable to leverage this to achieve a decent final blend similarity model. I’d break this down into a number of short-comings in my approach.

### Pre-Training Accuracy
Though I had access to strong pair prediction models as a result of my previous paper, I decided to reimplement these models from scratch. Simply put, this was a mistake. The hyperparameter optimization was nearly as long the second time around, and I had even less time than before. This led to marginal performance gains at best.  Still, I believe with more time and compute power, I likely could have produced a much stronger pre-trained model than in the original paper.

I did have new insights into the architecture, but I likely could have achieved better results from simply leveraging the existing MPNN-GNN model we released as part of that paper. 

In general, I think our field suffers quite strongly from this problem, as there are few public models with decent benchmark results, and researchers are generally forced to retrain models from scratch for every task. Hopefully, this challenge changes that.

### Separate Pre-Training and Fine-Tuning Pipelines
Because of the limited computing power and time I had available, I split the pre-training and fine-tuning pipelines into separate trials. This allowed me to focus on the long running pre-training trials, get reasonable results, and then evaluate many faster fine-tuning trials per model. Also, the pre-training trials could be pruned early, saving hours of compute time.

In the aroma-chemical pair paper, where the downstream fine-tuning task was a single-molecule odor-prediction task, this approach worked well, as the fine-tuning task was easier than the pre-training task. This meant a model that was good at pair prediction would be good at single-molecule prediction. This was not the case here, as the downstream task was significantly more challenging. In other words, a model that worked well at pair prediction might have been specialized to the fixed size of the pairs and could not transfer well to the more complicated varying blend cardinalities. 

Given more time and compute, I’d optimize these pipelines together, using the downstream task as the evaluation result. This means that hyperparameter trials could not be pruned at all, but would select for the most accurate models.

### Transfer Performance
During the cross-validation scheme, the best models were able to achieve decent predictions on certain folds, and failed on others. Further complicating this, the hardest fold varied from architecture to architecture. In general, the hardest dataset to predict was either Ravia or Bushdid, small models that tended to do well on the Bushdid fold (where the test set was larger than the combined training sets) failed to perform well on the Ravia fold (where the training set was significantly larger than the test set) which favored larger models.

Taking the mean or the median of these folds was not helpful, and though I tried to evaluate the models with standard deviations of performance below a certain threshold, the number of evaluation criteria became too large for me to tune in an exhaustive manner.

While an earlier model achieved a mean RMSE of 0.108 on this cross-validation scheme, it scores an RMSE of 0.138 on the leaderboard. Because I only submitted during the last few days, I did not have a chance to play around with a cross-validation scheme that would accurately predict the leaderboard performance.

As an additional regret, perhaps using the prior dataset carving scheme would have allowed me to discern which models actually transferred well to new data. While not carving the dataset allowed me to train on more pairs, it meant the evaluation task was significantly easier.

### Model Size
The final model during pre-training had around 350k parameters, which is significantly larger than the models we trained during the paper, and likely one of the largest in the competition.

It was exciting to train such a large model, and while I think I had the data available to train such a large model, I did not have the compute and engineering time to optimize the architecture and training pipeline. Even a single hyperparameter pre-training trial took about 90 minutes. It would likely have been better to focus on a model in the range of 50k parameters, for which trials could be completed in less than half an hour.

## Environmental Cost
Experiments were conducted using Google Cloud Platform in region us-east1, which has a carbon efficiency of 0.37 kgCO2/kWh. A cumulative of 234 hours of computation was performed on hardware of type A100 PCIe 40 (TDP of 250W) through Google Colab.

Total emissions are estimated to be 21.64 kgCO2 of which 100 percent was directly offset by the cloud provider.
Estimations were conducted using the MachineLearning Impact calculator, presented in Lacoste (2019).
