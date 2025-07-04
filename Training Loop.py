num_epochs = 100
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct = 0.0, 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
    
    train_losses.append(running_loss / len(train_loader.dataset))
    train_accuracies.append(correct / len(train_loader.dataset))
    
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        outputs = model(X_val_tensor)
        val_loss = criterion(outputs, y_val_tensor).item() * X_val_tensor.size(0)
        correct += ((torch.sigmoid(outputs) > 0.5).float() == y_val_tensor).sum().item()
    
    val_losses.append(val_loss / len(X_val_tensor))
    val_accuracies.append(correct / len(X_val_tensor))
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')
