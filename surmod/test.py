import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import surmod 



#! -----------------------------------------------------------------------

import test_functions as tfs
fun = tfs.rosenbrock

N = 100
xx, yy, f = fun(N)

N = 5
xt, yt, ft = fun(N)

# generate training sampling matrix and outputs
X_train = np.column_stack((xt.flatten(), yt.flatten()))
y_train = ft.flatten()

# generate testing sampling matrix and outputs
X_test = np.column_stack((xx.flatten(), yy.flatten()))
y_test = f.flatten()

print("Number of sampling points : {}".format(len(y_train)))

x_min = X_test.min(axis=0)
x_max = X_test.max(axis=0)

X_train = (X_train - X_test.min(axis=0))/(X_test.max(axis=0)-X_test.min(axis=0))
X_test = (X_test-X_test.min(axis=0))/(X_test.max(axis=0)-X_test.min(axis=0))

## Kriging model infil points : 

fun = lambda x: np.array([100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2])

varmin = np.array([-1, -1, 2, 2, 0, 0])
varmax = np.array([1, -1, 2, 2, 1, 1])
n_feat = 2
numiter = 5
numpop = 5
mu = 0.1
sigma = 0.2
child_factor = 2
gamma = 0.1
goal = 0


optim = structure(algorithm='GA', n_feat=n_feat, var_min=varmin, 
                  var_max=varmax, num_iter=numiter, 
                  num_pop=numpop, mu=mu, sigma=sigma, 
                  child_factor=2,  gamma=gamma, goal=goal)


x_opt = []
N_iter = 1

for ind in range(N_iter):
    krigger = surmod.rbf.Kriging(optim=optim, infill=True, verbose=False)

    x_new = krigger.fit(X_train,y_train)
    
    print(x_new)
    
    x_sc = x_new*(x_max-x_min) + x_min
    
    print(x_sc)
    
    y_new = fun(x_sc)
    
    X_train = np.row_stack((X_train,x_new))
    y_train = np.concatenate((y_train,y_new))
    
    x_opt.append(x_new)

varmin = np.array([-1, -1, 2, 2])
varmax = np.array([1, 1, 2, 2])

optim = structure(algorithm='GA', n_feat=n_feat, var_min=varmin, var_max=varmax, num_iter=numiter, num_pop=numpop, mu=mu, sigma=sigma, child_factor=2,  gamma=gamma)

krigger = surmod.rbf.Kriging(optim=optim, verbose=False)

krigger.fit(X_train,y_train)

print(krigger.theta)
print(krigger.parameters)

y_hat = krigger.predict(X_test)



#! ----------------------------------------------------------------------- 


# ofun = lambda x: np.sin(x) - np.sin(10/3*x)

# x = np.linspace(1,5,100)
# x_min = 1
# x_max = 5
# y = ofun(x)

# # Sample the function at 6 points :
# # xs = np.array([[1.5, 1.7, 2, 3, 3.5, 5]]).T
# xs = np.linspace(1.5,5,15)[:,None]
# ys = ofun(xs)

# # Scale xs :
# xs = (xs - x.min())/(x.max()-x.min())
# x = (x-x.min())/(x.max()-x.min())

# varmin = np.array([1,1])
# varmax = np.array([3,2])
# n_feat = 1
# numiter = 10
# numpop = 10
# mu = 0.1
# sigma = 0.1
# child_factor = 10
# gamma = 0.1

# optim = structure(n_feat=n_feat, var_min=varmin, var_max=varmax, num_iter=numiter, num_pop=numpop, mu=mu, sigma=sigma, child_factor=2,  gamma=gamma)

# krigger = surmod.rbf.Kriging(optim=optim, verbose=True)

# krigger.fit(xs,ys.ravel())