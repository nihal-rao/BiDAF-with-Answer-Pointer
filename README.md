# BiDAF with Answer-Pointer networks
An implementation of the [BiDAF](https://arxiv.org/abs/1611.01603) paper (with character level embeddings) for reading comprehension on the SqUAD v2 dataset. Also implemented an Answer Pointer network as per [Match-LSTM and Answer Pointer](https://arxiv.org/abs/1608.07905) to predict the final answer span. Achieves **78.83 F1 score, 69.45 EM score and 68% AvgNA score** on the dev dataset.

## Introduction
* Reading comprehension is a task in which a text passage (also called the context) is used to answer questions.
* The answers are a span (continous subsequence) of the context itself. Thus the model predicts start and end indices of the answer in the context.
* New to SQuAD v2, some questions are unanswerable ie., the context is insufficient to answer the question. The avgNA score of the model is the classification accuracy of the model in identifying such questions.
* This implementation uses the BiDAF model, with an Answer Pointer network as the predictor head instead of the Output layer (as shown below).
  
  ![](/images/bidaf-image.png)

* The answer pointer network predicts the end index conditioned on the start index of the answer.
* This is achieved by using the predicted start index probabilities as attention weights over the context.
* The aggregrated features are then used to predict the end position.
  
## Implementation details
* Only 325 of the total 442 articles in the SQuAD training dataset are used due to memory constraints.
* The answer pointer head uses dropout on the projected start pointer representation.
* A cosine learning rate scheduler is used.
* Hidden vectors in the BiDAF model are of 100 dimensions.
* Comparison of dev dataset negative log likelihood loss for this implementation (green) vs baseline model (blue)

 ![](/images/bidaf-image.png)

## Usage

* ```setup.py``` downloads and preprocesses the data.
* ```train.ipynb``` contains the training and logging code (Trained using kaggle gpu resources).

## Acknowledgements
* This repository uses the boilerplate code provided as part of the final project handout of the [Stanford CS224n : NLP with deep learning](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/) course.
