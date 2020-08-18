# D7047E
**Warning: There are security vulnerabilities in this repo, I will not fix them as this work was never intended to be secure, if you use any of this code in any project make sure you fix the security issues**

Repository for the course Advanced Deep Learning, D7047E, at Lulea University of Technology (LTU). This repo contains the exercise and project work done in the course. 

The log files for the exercises can be found in their respective folder, such exercsie 1 contains all log files for the experiments done in exercise 1. The main idea in the project is to try to viusalize what is happening inside an NLP similarily to what has been done with CNNs.

A longer description of the project can be found below:

## Project
In this work, we attempt to mirror  the  approach  of  producing  image  feature  visualization as  text  based  feature  visualization. The training process consists of optimizing  sentences  to  maximize  activations  in  a  pre-trained  model  for  text classification. 

We use a LSTM with pre-trained Word2Vec embeddings for sentiment analysis (SA) classification. The SA network will use pre-trained Word2Vec embeddings of dimension 300 and will then be trained to perform the downstream task of sentiment classification. We use the Word2Vec embeddings from [this](https://arxiv.org/abs/2003.11645). The output of the network will be binary such that it is either positive or negative. Once the network has been trained, we aim to visualize the features of different hidden layers by generating a random input, performing inference and then train the input such that we maximize the activations. We will start training on a small dataset, such as IMDB, and if need be we will increase the dataset size.

### Sources
 * https://distill.pub/2018/building-blocks/
 * https://distill.pub/2017/feature-visualization/
 * https://arxiv.org/abs/1506.01066
 * https://arxiv.org/abs/1602.08952
 * https://arxiv.org/abs/1809.07291
 * https://medium.com/@plusepsilon/visualizations-of-recurrent-neural-networks-c18f07779d56
 * https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
 * https://github.com/tosingithub/sdesk
