# INLP ASSIGNMENT 2

## Files
### model_1_train.py: 
- This file contains the code for training the **Feed Forward Neural Network**. 
- The model is trained on the training data and the trained model is saved in the file model.pth.
- Vocabularies are also saved in the file vocab.npy, upos_vocab.npy.

To run the file, use the following command:
```bash
python3 model_1_train.py
```
### model_1_tuning.py:
- This file contains the code for tuning the hyperparameters of the **Feed Forward Neural Network**. 

- The hyperparameters that are tuned are:
    - Activation function - ReLU, Tanh
    - Learning rate - 0.1, 0.01, 0.001
    - Number of hidden layers - 1, 2, 3
    - Hidden Dimension - 64, 128, 256
    - Embedding Dimension - 64, 128, 256
    - Epochs - 5, 10, 15, 20

To run the file, use the following command:
```bash
python3 model_1_tuning.py
```

### model_1_test.py:
- This file contains the code for testing the **Feed Forward Neural Network**.
- Loads the trained model from the file model.pth and tests it on the test data.
- Metrics used for evaluation are:
    - Accuracy
    - Precision (Macro and Micro)
    - Recall (Macro and Micro)
    - F1 Score (Macro and Micro)
    - Confusion Matrix

To run the file, use the following command:
```bash
python3 model_1_test.py
```
### model_2_train.py:
- This file contains the code for training the **LSTM** model.
- The model is trained on the training data and the trained model is saved in the file lstm_model.pth.
- Vocabularies are also saved in the file vocab.npy, upos_vocab.npy.

To run the file, use the following command:
```bash
python3 model_2_train.py
```

### model_2_tuning.py:
- This file contains the code for tuning the hyperparameters of the **LSTM** model.
- The hyperparameters that are tuned are:
    - Activation function - ReLU, Tanh
    - Learning rate - 0.1, 0.01, 0.001
    - Number of LSTM layers - 1, 2, 3
    - Hidden Dimension - 64, 128, 256
    - Embedding Dimension - 64, 128, 256
    - Epochs - 5, 10, 15
    - Bidirectional - True, False

To run the file, use the following command:
```bash
python3 model_2_tuning.py
```

### model_2_test.py:
- This file contains the code for testing the **LSTM** model.
- Loads the trained model from the file lstm_model.pth and tests it on the test data.
- Metrics used for evaluation are:
    - Accuracy
    - Precision (Macro and Micro)
    - Recall (Macro and Micro)
    - F1 Score (Macro and Micro)
    - Confusion Matrix

To run the file, use the following command:
```bash
python3 model_2_test.py
```
### utils.py:
- This file contains the code for the utility functions used in the training, tuning and testing of the models.

### epochs_vs_acc_graph.py:
- This file contains the code for plotting the graph of epochs vs accuracy and epochs vs loss for **LSTM** model on dev data.

To run the file, use the following command:
```bash
python3 epochs_vs_acc_graph.py
```
### p-s_graph.py:
- This file contains the code for plotting the graph of p vs loss and p vs accuracy for **FFNN** model on dev data.

To run the file, use the following command:
```bash
python3 p-s_graph.py
```
### pos_tagger.py:
- This file contains the code for the prediction of POS tags using the trained models given the input sentence.

To run the file, use the following command:
```bash
python3 pos_tagger.py -method <method>
```
- method: method to be used for prediction. It can be -f for FFFN or -r for LSTM.
- On running the file, it prompts the user to enter the sentence and then predicts the POS tags for the sentence.

### Report.pdf:
- This file contains the report for the assignment. It contains the details of the models, hyperparameters, results and the analysis of the results.


### Drive Link: 
[Drive link for saved models and vocabularies](https://drive.google.com/drive/folders/1bDIxznB2zjdUKChoQ7Hudy8ug3jBHo-I?usp=sharing)