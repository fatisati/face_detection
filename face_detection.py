from PIL import Image
from numpy import  array
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from sklearn import linear_model
from numpy import linalg as LA

x_train = []
y_train = []

x_test = []
y_test = []

images = []

with open('faces\\train.txt') as f:
    content = f.readlines()

with open('faces\\test.txt') as f:
    test_content = f.readlines()

avg = np.zeros((50, 50))

for line in content:
    path = line.split(' ')
    image_arr = array(Image.open(path[0]))
    x_train.append(image_arr)
    y_train.append(path[1])
    avg += image_arr


for line in test_content:
    path = line.split(' ')
    image_arr = array(Image.open(path[0]))
    x_test.append(image_arr)
    y_test.append(path[1])


avg /= len(x_train)
#toimage(avg).show()

#toimage(x_train[1]).show()

for i in range(len(x_train)):
    x_train[i] = x_train[i] - avg

for i in range(len(x_test)):
    x_test[i] = x_test[i] - avg


# print "avg"
# print avg

#toimage(x_train[1]).show()
# print "x"
# print x_train[1]

x = np.reshape(x_train, (540, 2500))

u, s, v = np.linalg.svd(x, full_matrices=False)

##showing v[i] eignface

# new_image = []
# vimage  = np.zeros((10, 2500))


# for i in range(10):
#
#     q = np.reshape(v[i], (50, 50))
#
#     new_image.append(q)
#     vimage[i] = v[i]
#
# print "done"
# new_im = toimage(np.reshape(vimage, (500, 50)))
# new_im.save('test.bmp')

#for i in range(0,1):
     #toimage(v[i]).show()
#     plt.imshow(v[i], cmap='gray')
#     plt.show()


x_r = []
err = []

#r approximation
# for r in range(1,200):
#     print r
#     sigma = np.zeros((540, 540))
#
#     for i in range(r):
#         sigma[i][i] = s[i]
#
#
#     tmp = np.matmul(u, sigma)
#     tmp2 = np.matmul(tmp, v)
#
#     err.append(LA.norm(tmp2 - x, 'fro'))
#     xr = np.reshape(tmp2[1], (50, 50))

    #toimage(xr).show()

        #isum += (np.abs(tmp - x_train[i])).sum() / (len(tmp)*len(tmp))
    #err.append(isum / 540)


# plt.plot(err)
# plt.show()
# # print err[80]

detection_err = []


r = 200
#for r in range(1, 201):
f = np.zeros((540, r))
a = np.reshape(x_train, (540, 2500))
b = np.reshape(v[0:r].T, (2500, r))

#print b[1]

np.matmul(a, b, f)


f_test = np.zeros((100, r))
a_test = np.reshape(x_test, (100, 2500))



np.matmul(a_test, b, f_test)



logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(f, y_train)

y_pred = logreg.predict(f_test)

sum = 0

for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        sum += 1

detection_err.append(sum)

print sum


# plt.plot(detection_err)
# plt.show()
