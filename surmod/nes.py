import numpy as np

def nes(fobj, optim):

    # hyperparameters
    npop = optim.num_pop # population size
    sigma = optim.sigma # noise standard deviation
    alpha = 0.01 # learning rate

    # start the optimization
    w = np.random.randn(optim.n_feat) # our initial guess is random
    r_best = fobj(w)

    for i in range(optim.num_iter):

        # print current fitness of the most likely parameter setting
        if i % 5 == 0:
            print('iter %d. w: %s, reward: %f' % 
                (i, str(w), fobj(w)))
  
        # initialize memory for a population of w's, and their rewards
        N = np.random.randn(npop, optim.n_feat) # samples from a normal distribution N(0,1)
        R = np.zeros(npop)
        w_try = w + sigma*N
        for j in range(npop):
            #   w_try = w + sigma*N[j] # jitter w using gaussian of sigma 0.1
            R[j] = fobj(w_try[j]) # evaluate the jittered version
  
        # Get best children :
        ind_best = np.argmin(R)
        if R[ind_best] < r_best:
            w = w_try[ind_best]
            r_best = R[ind_best]

        # standardize the rewards to have a gaussian distribution
        # A = (R - np.mean(R)) / np.std(R)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        # w = w + alpha/(npop*sigma) * np.dot(N.T, A)

    return w