import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA

mean1 = [10, 10]
mean2 = [22, 20]
cov = [[4, 4], [4, 9]]

class1 = np.random.multivariate_normal(mean1, cov, 1000).T
class2 = np.random.multivariate_normal(mean2, cov, 1000).T

# assert class1.shape == (2,1000), "The matrix has not the dimensions 3x20"
# assert class2.shape == (2,1000), "The matrix has not the dimensions 3x20"
plt.figure(1)

plt.plot(class1[0,:], class1[1,:], 'x')
plt.plot(class2[0,:], class2[1,:], 'x')



all_samples = np.concatenate((class1, class2), axis=1)
#assert all_samples.shape == (2,2000), "The matrix has not the dimensions 2x2000"
mlab_pca = mlabPCA(all_samples.T)


plt.figure(2)
plt.plot(mlab_pca.Y[0:1000,0],'o', markersize=7, color='blue', alpha=0.5, label='class1')

plt.plot(mlab_pca.Y[1000:2000,0],'^', markersize=7, color='red', alpha=0.5, label='class2')

#plt.axis('equal')

plt.figure(1)
sklearn_pca = sklearnPCA(n_components=1)
sklearn_transf = sklearn_pca.fit_transform(all_samples.T)


#plt.plot(sklearn_transf[0:1000,0], 'x')
#plt.plot(sklearn_transf[1000:2000,0], 'x')

#plt.plot(sklearn_transf[1000:2000,0],'^', markersize=7, color='red', alpha=0.5, label='class2')

proj = sklearn_pca.inverse_transform(sklearn_transf)
plt.figure(1)
plt.plot(proj[0:1000,0],proj[0:1000,1], 'x')
plt.plot(proj[1000:2000,0],proj[1000:2000,1], 'x')

loss = ((proj - all_samples.T) ** 2).mean()
print loss

plt.show()





