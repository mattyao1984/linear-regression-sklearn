import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 16)
model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
plt.scatter(y_test, predictions)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.show()

print("predictons:: ", predictions)