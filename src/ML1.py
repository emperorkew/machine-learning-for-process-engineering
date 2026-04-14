import numpy as np
from sklearn.decomposition import PCA

#transformation of variables is important to ensure equal importance of each variable in model training
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


