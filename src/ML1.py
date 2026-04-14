import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

#transformation of variables is important to ensure equal importance of each variable in model training
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

X = np.array([[1000, 0.01, 300],
              [1200, 0.06, 350],
              [1500, 0.1, 320]]
             )

scaler = StandardScaler().fit(X) #compute mean and std of each variable
X_scaled = scaler.transform(X) # transform X to have zero mean and unit variance


#min-max scaling as alternative
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#a lot of outliers -> median absolude devation from the mean (MAD)

#feature extraction from high-dimensional data with a lot of measuremments to lower dimensional
#data without losing a lot of information

features = PCA(1).fit_transform(X_scaled) #extracting one feature from the scaled measurements


#reading raw quadratic data
data = np.loadtxt('../data/quadratic_data.csv', delimiter=',')
x = data[:, 0, None]
y = data[:, 1, None]

#plot data in cartesian field
plt.scatter(x, y)
plt.show()

#generate quadratic features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x)

#scale model inputs
poly_scaler = StandardScaler()
X_poly_scaled = poly_scaler.fit_transform(X_poly)

#fit linear model & predict
model = LinearRegression()
model.fit(X_poly_scaled, y)
y_pred = model.predict(X_poly_scaled)

#show prediction
plt.plot(x, y_pred, color='red')
plt.scatter(x, y)
plt.show()