

# Next Word Prediction using LSTM

This project is inspired by the **LGM Data Science Virtual Internship** assignment. The goal is to build a simplified version of GPT-style models that predict the **next word** in a sentence using deep learning.

##  Project Overview

* Dataset: *The Adventures of Sherlock Holmes* (Project Gutenberg) or any large text file (`book.txt`).
* Input: A sequence of words (e.g., 5 words).
* Output: The model predicts the most probable next word.
* Model: **Embedding + LSTM + Dense layers** trained on text data.


## Tech Stack

* Python 3.10+
* TensorFlow / Keras
* NumPy, Matplotlib
* Google Colab / Jupyter Notebook

## Steps Implemented

1. **Load dataset** (Sherlock Holmes text or custom `book.txt`).
2. **Preprocess text** (lowercase, tokenize, sequences).
3. **Create training data** (X = words, y = next word).
4. **Build model**

   * Embedding layer
   * LSTM with dropout
   * Dense softmax layer
5. **Train with EarlyStopping** to prevent overfitting.
6. **Plot training curves** (Accuracy & Loss).
7. **Predict next word** given custom input.



## Training Results

Example training curves:

<img width="1702" height="617" alt="image" src="https://github.com/user-attachments/assets/e40b140b-d32f-4d46-a296-2b192dd51ecb" />


* Accuracy rises steadily for training and validation.
* Loss decreases for training, but validation loss shows overfitting.
* With more data (full book), better embeddings, and dropout, the model improves.



##  Example Prediction

```python
Input: "sherlock holmes was sitting in"  
Predicted Next Word: "the"
```



## Future Improvements

* Use **entire text** instead of limited words.
* Add **stacked LSTMs** or GRU layers.
* Try **pre-trained embeddings (GloVe / Word2Vec)**.
* Generate **multiple words** instead of one.



