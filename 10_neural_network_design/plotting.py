import matplotlib.pyplot as plt
import numpy as np

labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
accuracy = [0.738, 0.823, 0.309, 0.329, 0.492, 0.446, 0.619, 0.726, 0.471, 0.479]
misclassification = np.ones(len(accuracy)) - accuracy
plt.bar(labels, misclassification)
plt.title('Misclassification Rate by Category')
plt.show()