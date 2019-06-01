import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import model_main as m

importances = m.clf.feature_importances_[:18]
indices = np.argsort(importances)

print("Feature ranking:")

plt.figure(1)
plt.title("Feature importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), m.feature_names[indices])
plt.xlabel('Relative Importance')
plt.show()

