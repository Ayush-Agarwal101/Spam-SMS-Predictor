# 📩 Spam SMS Detector (Neural Network)

This project implements a **Spam/Ham SMS Classifier** using a fully custom neural network built from scratch with **NumPy**, without any high-level deep learning frameworks like TensorFlow or PyTorch. It processes raw SMS messages, vectorizes them, and trains a neural network to classify messages as spam or not spam.

---

## 🚀 Features

- Feedforward Neural Network with:
  - ReLU activation for hidden layers
  - Sigmoid activation for output
  - Binary Cross-Entropy loss
- Manual backpropagation and gradient update logic
- Weight and Bias encapsulated as separate classes
- Count-based vectorization using `CountVectorizer`
- Feature scaling using `MinMaxScaler`
- Training and testing on user-defined batch sizes
- Live prediction: enter your own message and check if it's spam
- Cost-vs-Epoch graph plotted using `matplotlib`

---

## 📁 Dataset

Dataset used: [SMS Spam Collection Dataset](https://www.kaggle.com/code/karnikakapoor/spam-or-ham-sms-classifier/input)

- File: `spam.csv`
- Source: UCI ML Repository (via Kaggle)
- Used columns:
  - `v1`: Label (`ham` or `spam`)
  - `v2`: Message text

📝 Make sure to download `spam.csv` and place it in the root of your project folder.

---

## 🌐 Language Support

This model is currently trained exclusively on **English-language SMS messages**, as the vocabulary is built from English words in the dataset.  
To support other languages, you can **replace or extend** the dataset with a sufficient number of labeled messages in the desired language. The model will then rebuild the vocabulary accordingly during preprocessing.

---

## 📦 Requirements

Install the following Python libraries:

```bash
%pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ How to Run

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/spam-sms-detector.git
   cd spam-sms-detector
   ```

2. **Download the dataset**:
   - Go to: https://www.kaggle.com/code/karnikakapoor/spam-or-ham-sms-classifier/input
   - Download `spam.csv`
   - Place the file in your project directory

3. **Open the notebook**:
   ```bash
   jupyter notebook "Spam SMS Detector.ipynb"
   ```
   or if you're using VS Code or JupyterLab, open it directly from the GUI.

4. **Run all cells**
   Click Run All or run cell-by-cell and follow the prompts. Input required batch sizes when prompted:
   - Training batch size
   - Testing batch size

5. **Start classifying SMS messages**!

---

## 🧠 Network Architecture

```
Input Layer:    (based on vocabulary size)
Hidden Layer:   32 neurons (ReLU)
Output Layer:   1 neuron (Sigmoid)
```

---

## 📊 Output Sample

```
Enter training batch size: 4000
Enter testing batch size: 1000
Training on 4000 samples with 100 epochs...
Epoch 1, Cost: 0.6391
Epoch 10, Cost: 0.4712
...

Testing on 1000 samples...
Accuracy on test set: 96.10%

Enter a message to check if spam or not: Congratulations! You’ve won a free iPhone!
Spam
```

---

## 🧑‍💻 Author

**Ayush Agarwal**

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
