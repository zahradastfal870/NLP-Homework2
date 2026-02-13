import numpy as np

# Confusion matrix
# Rows = System prediction
# Columns = Gold labels
#        Cat  Dog  Rabbit
conf_matrix = np.array([
    [5, 10, 5],     # Predicted Cat
    [15, 20, 10],   # Predicted Dog
    [0, 15, 10]     # Predicted Rabbit
])

classes = ["Cat", "Dog", "Rabbit"]

# ---- Per-class Precision and Recall ----
precision = []
recall = []

for i in range(len(classes)):
    TP = conf_matrix[i, i]
    predicted_total = np.sum(conf_matrix[i, :])
    actual_total = np.sum(conf_matrix[:, i])
    
    p = TP / predicted_total
    r = TP / actual_total
    
    precision.append(p)
    recall.append(r)
    
    print(f"{classes[i]}:")
    print(f"  Precision = {p:.4f}")
    print(f"  Recall    = {r:.4f}")
    print()

# ---- Macro Average ----
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)

print("Macro-Averaged:")
print(f"  Precision = {macro_precision:.4f}")
print(f"  Recall    = {macro_recall:.4f}")
print()

# ---- Micro Average ----
TP_total = np.trace(conf_matrix)
total = np.sum(conf_matrix)

micro_precision = TP_total / total
micro_recall = TP_total / total

print("Micro-Averaged:")
print(f"  Precision = {micro_precision:.4f}")
print(f"  Recall    = {micro_recall:.4f}")
