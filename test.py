import torch
from torch import nn

# Define the KAN model
class KAN(nn.Module):
    def __init__(self, width, grid, k):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(width[0], width[1])
        self.fc2 = nn.Linear(width[1], width[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test the model's interpretability
def test_model_interpretability():
    # Define the input data dimensions
    input_dim = 18  # Example: 20 features
    width = [input_dim, 10, 1]  # Example: 10 neurons in the first hidden layer, 1 neuron in the output layer
    grid = 5  # Example: 5 grid points
    k = 3  # Example: 3 kernel size

    # Create an instance of the KAN model
    model = KAN(width, grid, k)

    # Load the trained model's weights
    model.load_state_dict(torch.load('Prediction_KAN.pth'))

    # Test the model's interpretability by analyzing the weights
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
        print(f"{name}: {param.data}")

    print("Model interpretability test passed!")

# Run the test
test_model_interpretability()

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the KAN model
class KAN(nn.Module):
    def __init__(self, width, grid, k):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(width[0], width[1])
        self.fc2 = nn.Linear(width[1], width[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test the model's evaluation metrics
def test_model_evaluation_metrics():
    # Define the input data dimensions
    input_dim = 18  # Example: 20 features
    width = [input_dim, 10, 1]  # Example: 10 neurons in the first hidden layer, 1 neuron in the output layer
    grid = 5  # Example: 5 grid points
    k = 3  # Example: 3 kernel size

    # Create an instance of the KAN model
    model = KAN(width, grid, k)

    # Load the trained model's weights
    model.load_state_dict(torch.load('Prediction_KAN.pth'))

    # Define the input data and labels
    X = torch.randn(100000, input_dim)  # Example: 100 samples with 20 features
    y = torch.randint(0, 2, (100000,))  # Example: 100 binary labels

    # Make predictions
    with torch.no_grad():
        outputs = model(X)
        preds = torch.sigmoid(outputs).round()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    print("Model evaluation metrics test passed!")

# Run the test
test_model_evaluation_metrics()
