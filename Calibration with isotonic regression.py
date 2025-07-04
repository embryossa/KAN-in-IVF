
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

X_val_tensor = dataset['val_input']
y_val_tensor = dataset['val_label']
# Получаем некалиброванные выходы модели
with torch.no_grad():
    outputs = model(X_val_tensor)
    uncalibrated_probs = torch.sigmoid(outputs).cpu().numpy()

# Применяем изотоническую регрессию для калибровки
ir = IsotonicRegression(out_of_bounds='clip')
calibrated_probs = ir.fit_transform(uncalibrated_probs.ravel(), y_val_tensor.cpu().numpy().ravel())

# Строим calibration plot
fraction_of_positives, mean_predicted_value = calibration_curve(y_val_tensor.cpu().numpy(), uncalibrated_probs, n_bins=10)

plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Uncalibrated')

fraction_of_positives, mean_predicted_value = calibration_curve(y_val_tensor.cpu().numpy(), calibrated_probs, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, marker='s', label='Calibrated')

plt.legend()
plt.xlabel('Mean predicted value')
plt.ylabel('Fraction of positives')
plt.title('Calibration Plot')
plt.show()
