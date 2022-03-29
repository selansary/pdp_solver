# import numpy as np
# import matplotlib.pyplot as plt
# data = [[30, 25, 50, 20],
# [40, 23, 51, 17],
# [35, 22, 45, 19]]



# X = np.arange(4)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# Set the width and height of the figure
# plt.figure(figsize=(10,6))
fig = plt.figure()

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=[5, 7, 9, 11], y=[0.48, 2.67, 50.02, 500])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")

fig.show()
