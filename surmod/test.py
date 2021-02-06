import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import surmod 

ofun = lambda x: np.sin(x) - np.sin(10/3*x)

x = np.linspace(1,5,100)
x_min = 1
x_max = 5
y = ofun(x)

# Sample the function at 6 points :
# xs = np.array([[1.5, 1.7, 2, 3, 3.5, 5]]).T
xs = np.linspace(1.5,5,15)[:,None]
ys = ofun(xs)

# Scale xs :
xs = (xs - x.min())/(x.max()-x.min())
x = (x-x.min())/(x.max()-x.min())

varmin = np.array([1,1])
varmax = np.array([3,2])
n_feat = 1
numiter = 10
numpop = 10
mu = 0.1
sigma = 0.1
child_factor = 10
gamma = 0.1

optim = structure(n_feat=n_feat, var_min=varmin, var_max=varmax, num_iter=numiter, num_pop=numpop, mu=mu, sigma=sigma, child_factor=2,  gamma=gamma)

krigger = surmod.rbf.Kriging(optim=optim, verbose=True)

krigger.fit(xs,ys.ravel())