Here's an edited version of your GitHub README file that is more structured and includes all necessary details:

---

# KAN-in-IVF

## Introducing Kolmogorov-Arnold Networks (KANs) in IVF Treatment Prediction

Kolmogorov-Arnold Networks (KANs) are a promising alternative to traditional Multi-Layer Perceptrons (MLPs). While MLPs are rooted in the universal approximation theorem, KANs are based on the Kolmogorov-Arnold representation theorem. This fundamental difference gives KANs several advantages in terms of model accuracy, interpretability, and efficiency.

## KAN Model Training and Evaluation

This script trains and evaluates a neural network model (KAN) for predicting outcomes in IVF procedures. The script includes loading the model, training it on new data, and evaluating its performance using various metrics. Additionally, it plots training curves and evaluation metrics such as ROC and Precision-Recall curves.

### Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch pandas scikit-learn matplotlib openpyxl
```

Ensure you have the following versions:

```bash
python==3.9.7
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
```

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/kan-model.git
cd kan-model
```

### Usage

#### Load and Define the Model

The `KAN` class defines a simple neural network model with one hidden layer. The model is loaded with pre-trained weights.

#### Load and Preprocess Data

Load data from an Excel file, select relevant features, and preprocess them using `StandardScaler`.

#### Create DataLoader

Create `DataLoader` for the training data.

#### Define Loss and Optimizer

Define the loss function and the optimizer for training.

#### Training Loop

Train the model and validate it on the validation set. Track training and validation losses.

#### Save the Model

Save the trained model.

```python
torch.save(model.state_dict(), 'updated_model.pth')
```

#### Evaluate on Test Data

Evaluate the model on the test data and compute metrics.

### Conclusion

This script provides a comprehensive workflow for training and evaluating a neural network model on IVF data. By following the steps outlined above, you can load your data, preprocess it, train the model, evaluate its performance, and visualize the results.

The integration of KANs into our predictive models highlights our commitment to innovation and excellence in reproductive medicine. We believe this will significantly enhance our predictive capabilities and, ultimately, improve patient outcomes.

### Citation

The original Paper for KAN research:

```plaintext
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang,
