# [TODO] Rename this file to [your student ID].py

# DO NOT EDIT THESE IMPORT STATEMENTS!
from cmath import pi
import math
from nntplib import NNTP
from telnetlib import GA
from tracemalloc import stop
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from time import sleep
##########

###########
# DO NOT edit this function
def EM(gmm):
    """
    Runs the expectation-maximization algorithm on a GMM
    
    Input: 
    gmm (Class GMM): our GMM instance
    
    Returns: 
    Nothing, but it should modify gmm using the previously defined functions
    """
    
    #Log likelihood computation
    if gmm.verbose:
        print('Iteration: {:4d}'.format(0), flush = True)

    # Compute mixture normalization for all the samples
    normalization(gmm)

    # Compute initial Log likelihoods
    logLikelihood(gmm)
          
    # Repeat EM iterations
    for n in range(1,gmm.max_iter):               
        # Expectation step
        expectation(gmm)

        # Maximization step
        maximization(gmm)
        
        # Update mixture normalization for all the samples
        normalization(gmm)
        
        # Update the Log likelihood estimate
        logLikelihood(gmm)

        # Logging and plotting
        if gmm.verbose:
            print('Iteration: {:4d} - log likelihood: {:1.6f}'.format(n, gmm.log_likelihoods[-1]), flush = True)
        
        if gmm.do_plot:
            gmm.plotGMM(ellipse = True)
            plt.pause(0.05)
            sleep(0.15)
            if n != gmm.max_iter - 1:
                plt.close()
            
        # Compute the relative log-likelihood improvement and claim victory if a convergence tolerance is met
        relative_error = abs(gmm.log_likelihoods[-2] / gmm.log_likelihoods[-1])
        if (abs(1 - relative_error) < gmm.tol):
            expectation(gmm)
            if gmm.verbose:
                print('SUCCESS: Your EM process converged.', flush = True)
            return

    plt.show()

    if gmm.verbose:
        print('ERROR: You ran out of iterations before converging.', flush = True)
###########



# [TODO] Deliverable 1: Polynomial regression with MLE
def PolRegMLE(p, mu, x, y ):
    """ 
    Function that computes the MLE linear system matrix A and vector b 
    
    Inputs: 
    p: number of parameters / polynomial coefficients / degree of polynomial + 1
    mu: mean of the assumed Gaussian noise
    x (np.array(n,)): independent data variables
    y (np.array(n,)): dependent data variables

    Returns: 
    A, b (tuple(np.array((p,p)), np.array(p))), 
    Return the matrix A and vector b of the MLE linear system
    """
    ### BEGIN SOLUTION
    A = np.zeros((p,p))
    b = np.zeros(p)
    for i in range(p):
        for j in range(p):
            Xs = 0
            for s in range(x.shape[0]):
                Xs = Xs + (x[s]**(i+j))
            A[i][j] = Xs
    for i in range(p):
        sum = 0
        for s in range(x.shape[0]):
            sum = sum+((x[s]**i)*(y[s]-mu))
        b[i] = sum
   
    
    ### END SOLUTION
    return A, b


# [TODO] Deliverable 2: Polynomial regression with MAP
def PolRegMAP(p, mu_noise, sigma_noise, mu_params, sigma_params, x, y):
    """
    Function that computes the MAP linear system matrix A and vector b 
        
    Inputs: 
    p: number of parameters
    mu_noise: mean of the Gaussian noise
    sigma_noise: standard deviation of the Gaussian noise
    mu_params: mean of the parameters' Gaussian prior
    sigma_params: standard deviation of the parameters' Gaussian prior
    x (np.array(n,)): independent data variables
    y (np.array(n,)): dependent data variables

    Returns: 
    A, b (tuple(np.array((p,p)), np.array(p))), 
    Return the matrix A and vector b of the MAP linear system
    """

    A = np.zeros((p,p))
    b = np.zeros(p)
    ### BEGIN SOLUTION
    for i in range(p):
        for j in range(p):
            aSum = 0
            if(i == j):
                for s in range(x.shape[0]):
                    aSum = aSum + ((x[s]**(i+j)))
                aSum = aSum +((sigma_noise**2)/(sigma_params**2))
            else:
                for s in range(x.shape[0]):
                    aSum = aSum + (x[s]**(i+j))
            A[i][j] = aSum
    for i in range(p):
        aSum = 0
        for s in range(x.shape[0]):
            aSum = aSum + ((x[s]**i)*(y[s]-mu_noise))
        b[i] = aSum+(mu_params*((sigma_noise**2)/(sigma_params**2)))
    
    ### END SOLUTION
    return A, b



# [TODO] Deliverable 3: Polynomial regression with LLS
def PolRegLLS(p, n, x, y):
    """ 
    Function that compute the LLS linear system matrix A and vector b 
        
    Inputs: 
    p: number of parameters
    n: number of samples
    x (np.array(n,)): independent sample variables
    y (np.array(n,)): dependent sample variables

    Returns: 
    A, b (tuple(np.array((p,p)), np.array(p))), 
    Return the matrix A and vector b of the linear system
    """
    
    A = np.zeros((p,p))
    b = np.zeros(p)

    ### BEGIN SOLUTION
    result = PolRegMLE(p,0,x,y)
    A = result[0]
    b = result[1]
    At = A.transpose()@A
    b = A.transpose()@b
    ### END SOLUTION
    return (At, b)

###########
# DO NOT EDIT
# Class that encapsulates the Gaussian mixture model data and utility methods
class GMM:
    def __init__( self, X, n_components = 10, reg_covar = 1e-2, tol = 1e-4, 
                  max_iter = 100, verbose = True, do_plot = False, mu_init = None):
        """
        Constructor of the GMM class
            
        Inputs: 
        X (np.array((n_samples, n_dim))): array containing the n_samples n_dim-dimensional data samples
        n_components (int): number of mixture components for our GMM
        reg_covar (float): regularization value to add to the diagonal of the covariance matrices
        tol (float): relative log-likelihood tolerance, at which point we will terminate the iterative EM algorithm
        max_iter (int): maximum number of iterations, after which we terminate the iterative EM algorithm
        mu_init (np.array((n_components, n_dim))): array containing initial means for each Gaussian in the mixture; if None, we will sample them from X.
        verbose (bool): True to print verbose output
        do_plot (bool): True to plot GMM evolution after each EM iteration
        """

        self.X = X.astype(np.float32)
        self.n_samples, self.n_dim = self.X.shape
        self.n_components = n_components
        self.reg_covar = reg_covar**2
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.do_plot = do_plot
        self.reg_covar = reg_covar**2
        
        # regularization matrix
        self.reg_cov = self.reg_covar * np.identity(self.n_dim, dtype = np.float32)
        
        # initial (isotropic) covariance extent
        self.init_covar = 0.5 * (np.amax(X) - np.amin(X)) / self.n_components          
        
        # initial covariance matrix
        self.init_cov = self.init_covar * np.identity(self.n_dim, dtype = np.float32) 
                
        
        # Initialize the mu, covariance and pi values
        if mu_init is None:
            # Initialize mean vector as random element of X
            self.mu = self.X[np.random.choice(range(0,self.n_samples), self.n_components, replace=False),:]
        else:
            try:
                assert( mu_init.shape[0] == self.n_components and mu_init.shape[1] == self.n_dim )
            except:
                raise Exception('Can\'t plot if not 2D')
            
            # Initialize mean vector from the provided means mu_init
            self.mu = mu_init 
        
        # Initialize covariances as diagonal matrices (isotropic Gaussians)
        self.cov = np.zeros((self.n_components, self.n_dim, self.n_dim), dtype=np.float32)
        for c in range(self.n_components):
            self.cov[c,:,:] = self.init_cov

        # Python list of the n_components multivariate Gaussian distributions
        # The .pdf method of the Gaussian's allows you to evaluate them at a vector of input locations
        self.gauss = []
        for c in range(self.n_components):
            self.gauss.append( multivariate_normal( mean = self.mu[c,:], 
                                                    cov = self.cov[c,:,:]) )
        
        # Probabilities of selecting a specific Gaussian from the mixture
        # Initialized to uniform probability for selecting each Gaussian, i.e., 1/K
        self.pi = np.full(self.n_components, 1./self.n_components, dtype = np.float32)
        
        # The weight of each Gaussian in the mixture
        # Initialized to 0
        self.weight = np.zeros(self.n_components, dtype = np.float32)
        
        # The probabilities of sample X_i belonging to Gaussian N_c
        # Initialized to 0
        self.alpha = np.zeros((self.n_samples, self.n_components), dtype = np.float32)
        
        # Normalization for alpha
        # Initialized to 0
        self.beta = np.zeros(self.n_samples)
        
        # Latent labels (indices) of the Gaussian with maximum probability of having generated sample X_i
        # Initialized to 0
        self.Z = np.zeros(self.n_samples, dtype = np.int32)
        
        # Python list for logging the log-likelihood after each iteration of the EM algorithm
        self.log_likelihoods = [] 

    # Some visualization helper routines
    def draw_ellipse(self, k, **kwargs):
        # Draw an ellipse corresponding to the k-th Gaussian
        try:
            assert(self.n_dim == 2)
        except:
            raise Exception('Can\'t plot if not 2D')
            
        ax = plt.gca()

        # Convert covariance to principal axes
        U, s, Vt = np.linalg.svd(self.cov[k,:,:])
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(self.mu[k,:], nsig * width, nsig * height, angle, **kwargs))
    
    # Plot the mixture
    def plotGMM(self, samples = None, labels = None, ellipse = True):
        try:
            assert(self.n_dim == 2)
        except:
            raise Exception('Can\'t plot if not 2D')
            
        plt.figure(figsize=(10,10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_components))
        if samples is None or labels is None:
            plt.scatter(self.X[:, 0], self.X[:, 1], c=colors[self.Z,:], s=10)
        else:
            try:
                assert(self.n_dim == samples.shape[1] and samples.shape[0]  == labels.shape[0])
            except:
                raise Exception('Can\'t plot if not 2D')
            plt.scatter(samples[:, 0], samples[:, 1], c=colors[labels,:], s=10)
        plt.axis('equal')
        plt.axis('tight')
        
        w_factor = 0.2 / self.weight.max()
        if ellipse:
            for k in range(self.n_components):
                self.draw_ellipse(k, alpha = w_factor * self.weight[k], color = colors[k,:3])
                plt.scatter(self.mu[k,0], self.mu[k,1], marker = '*', s = 100)
#############



# [TODO] Deliverable 4: Computing the mixture normalization
def normalization(gmm:GMM):     
    """
    Compute the mixture normalization factor for all the data samples

    Input: 
    gmm (Class GMM): our GMM instance

    Returns: 
    Nothing, but you should modify gmm.beta
    """

    ### BEGIN SOLUTION
    array = np.zeros((gmm.n_components,gmm.n_samples))
    for c in range(gmm.n_components):
        foo = multivariate_normal(mean=gmm.mu[c],cov=gmm.cov[c])
        N = foo.pdf(gmm.X)*gmm.pi[c]
        array[c] = N
    #print("theN: ",array)
    gmm.beta = array.sum(axis=0)

        #sum = sum+(gmm.pi[c]*N)
        #print("test: ", sum)

    ### END SOLUTION
    #print("gmm.beta: ",gmm.beta)


# [TODO] Deliverable 5: E-Step
def expectation(gmm:GMM):           
    """
    The expectation step

    Input:
    gmm (Class GMM): our GMM instance

    Returns: 
    Nothing, but you should modify gmm.alpha
    """

    ### BEGIN SOLUTION
    array = np.zeros((gmm.n_components,gmm.n_samples))
    for j in range(gmm.n_components):
        foo = multivariate_normal(mean=gmm.mu[j],cov=gmm.cov[j])
        N = foo.pdf(gmm.X)
        a = ((gmm.pi[j]*N)/gmm.beta)
        array[j] = a
    array = array.transpose()
        
    #print("the_A: ",array)
    gmm.alpha = array
    #print(gmm.alpha)

    ### END SOLUTION



# [TODO] Deliverable 6: M-Step
def maximization(gmm:GMM):                   
    """
    The maximization step
    
    Input: 
    gmm (Class GMM): our GMM instance
    
    Returns: 
    Nothing, but you should modify gmm.Z, gmm.weight, gmm.pi, gmm.mu, gmm.cov, and gmm.gauss    
    """
    
    # You can loop over the mixture components ONLY
    # and assume that you already know alpha
    # Hint 1: np.argmax is useful, here
    # Hint 2: don't forgot to regularize your covariance matrices with gmm.reg_cov
    
    ### BEGIN SOLUTION
    # print("sahpe z: ",gmm.Z.shape)
    # print("shpae a:", gmm.alpha.shape)
    argmaxA = np.argmax(gmm.alpha,axis=1)
    #print(argmaxA.shape)
    #for i in range(gmm.n_samples):
    gmm.Z = argmaxA
    # print("shpae a: ", gmm.alpha.shape)
    # print("shape w: ", gmm.weight.shape)
    sumW = gmm.alpha.sum(axis=0)
    #print(sumW.shape)
    #for j in range(gmm.n_components):
    gmm.weight = sumW
    #print(gmm.weight)
    #for j in range(gmm.n_components):
    gmm.pi = gmm.weight/gmm.n_samples
    #print("pi: ",gmm.pi)
    for j in range(gmm.n_components):
        foo = (gmm.alpha[:,j])
    
        print("foo: ",foo)
        exit()
        #gmm.mu[j] = (1/gmm.weight[j])*sum
    
    for j in range(gmm.n_components):
        sum = 0
        for i in range(gmm.n_samples):
            xsu = (gmm.X[i] - gmm.mu[j])[np.newaxis]
            transXsu =  np.transpose((gmm.X[i] - gmm.mu[j])[np.newaxis])
            sum = sum+ (gmm.alpha[i][j]*(transXsu@xsu))
        gmm.cov[j] = (sum+gmm.reg_cov)*(1/gmm.weight[j])
    Gaussians = []
    for i in range(gmm.n_components):
        Gaussians.append([multivariate_normal( mean = gmm.mu[i] , cov = gmm.cov[i])])
    gmm.gauss = Gaussians
    ### END SOLUTION


# [TODO] Deliverable 7: Compute the log-likelihood
def logLikelihood(gmm:GMM):                        
    """
    Log-likelihood computation

    Input: 
    gmm (Class GMM): our GMM instance

    Returns: 
    Nothing, but you should modify gmm.log_likelihoods
    """

    # Note: you need to append to gmm.log_likelihoods
    
    ### BEGIN SOLUTION
    #print("betas: ",gmm.beta)
    # sum = 0
    # for i in range(gmm.n_samples):
    #     sum = sum+np.log(gmm.beta[i])
    sum = (np.log(gmm.beta)).sum()
    gmm.log_likelihoods.append(sum)
    #print(sum)
    #print("likelyhood: ",gmm.log_likelihoods)
    ### END SOLUTION



# Some example test routines for the deliverables. 
# Feel free to write and include your own tests here.
# Code in this main block will not count for credit, 
# but the collaboration and plagiarism policies still hold.
if __name__ == '__main__':
    # ! generate noisy polynomial data to test your estimators
    np.random.seed(0)
    n = 40     # Number of samples
    m = 7      # Degree of the polynomial
    p = m + 1  # Number of parameters 

    # Generate random polynomial coefficients ~ N(1,2)
    mu_params = 1
    sigma_params = 2
    theta = np.random.normal(mu_params, sigma_params, p)
    poly = np.poly1d(theta[::-1])

    # Model the sample noise as ~ N(-10,10)
    mu_noise = -10.   # Mean of the noise distribution
    sigma_noise = 10. # Standard deviation of the noise distribution
    epsilon = np.random.normal(mu_noise, sigma_noise, n)

    # Generate our noisy polynomial data points
    x = np.linspace(-1.5, 1.5, n)
    y = poly(x) + epsilon

    
    # ! Generate your MLE, MAP and LLS linear systems
    A_MLE, b_MLE = PolRegMLE(p, mu_noise, x, y)
    A_MAP, b_MAP = PolRegMAP(p, mu_noise, sigma_noise, mu_params, sigma_params, x, y)
    A_LLS, b_LLS = PolRegLLS(p, n, x, y)

    # Solve the systems
    theta_MLE = np.linalg.solve(A_MLE, b_MLE)
    theta_MAP = np.linalg.solve(A_MAP, b_MAP)
    theta_LLS = np.linalg.solve(A_LLS, b_LLS)

    
    # ! MAP should be more robust than MLE and LLS since it is given the exact prior. Also, MLE and LLS should look quite similar.

    # Plot polynomial coefficients and errors
    fig, axes = plt.subplots()
    plt.title("Estimated polynomial parameter values and errors")
    plt.ylabel("coefficient values")
    plt.xlabel("coefficient indices")
    axes.bar(np.arange(p), theta, 0.125, label=r"$\theta$"), 
    axes.bar(np.arange(p) + 0.125, theta_MLE, 0.125, label=r"$\theta_{MLE}$ error: " + str(round( np.linalg.norm(theta_MLE-theta,2), 2) )), 
    axes.bar(np.arange(p) + 0.250, theta_MAP, 0.125, label=r"$\theta_{MAP}$ error: " + str(round( np.linalg.norm(theta_MAP-theta,2), 2) )),
    axes.bar(np.arange(p) + 0.375, theta_LLS, 0.125, label=r"$\theta_{LLS}$ error: " + str(round( np.linalg.norm(theta_LLS-theta,2), 2) ))
    axes.legend()
    plt.show() # This is a blocking call! Close the plotting window to continue execution flow.


    # Plot polynomial reconstructions and sampled data
    z = np.linspace(-2., 1.75, 100) # plot more densely
    plt.figure()
    plt.title('Noisy polynomial data and fits')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.scatter(x, y, label='data: $(x, y)$')
    plt.plot(z, poly(z), label='$p_{{ {0} }}(x)$'.format(m))
    plt.plot(z, np.polyval(theta_MLE[::-1], z), '--',label='$p_{{ {0} }} (x | \\theta_{{MLE}})$'.format(m))
    plt.plot(z, np.polyval(theta_MAP[::-1], z), '--',label='$p_{{ {0} }} (x | \\theta_{{MAP}})$'.format(m))
    plt.plot(z, np.polyval(theta_LLS[::-1], z), '--',label='$p_{{ {0} }} (x | \\theta_{{LLS}})$'.format(m))
    plt.legend()
    plt.show() # This is a blocking call! Close the plotting window to continue execution flow.



    # ! Experiment with your EM implementation
    # Let's draw samples from a mixture of four (4) Gaussians
    # and then fit a 4-component GMM to the data using EM
    np.random.seed(0)
    K = 4
    n = 500 * K # 500 samples per mixture
    mus = np.array([[0.25, 0.25],
                    [0.75, 0.75],
                    [0.25, 0.75],
                    [0.75, 0.25]])
    
    Sigmas = np.array([ np.array([0.0002]) * [[75, -70], [-70, 75]],
                        np.array([0.0002]) * [[1, 0], [0, 75]],
                        np.array([0.0200]) * [[1, 0], [0, 1]],
                        np.array([0.0100]) * [[1, 0], [0, 1]] ])
    
    Gaussians = [ multivariate_normal( mean = mus[0,:] , cov = Sigmas[0,:,:]), 
                  multivariate_normal( mean = mus[1,:] , cov = Sigmas[1,:,:]), 
                  multivariate_normal( mean = mus[2,:] , cov = Sigmas[2,:,:]), 
                  multivariate_normal( mean = mus[3,:] , cov = Sigmas[3,:,:]) ]

    data = [Gaussians[int(i / (n/K))].rvs() for i in range(n)]
    data = np.array(data)

    # set do_plot = False if you don't want plotting after every iteration of the EM algorithm
    gmm_test  = GMM( data, K, reg_covar = 1e-3, 
                     tol = 1e-6, max_iter = 100, 
                     verbose = True, do_plot = True) 

    #print("gmm_test_mu: ", gmm_test.mu)
    # print("gmm_test_pis: ", gmm_test.pi)
    #print("gmm_test_cov: ",gmm_test.cov.shape)
    # print("gmm_test_x: ",gmm_test.X)
    #print("gmm_Test_cov_shape: ",gmm_test.cov)
    # print("d: ",gmm_test.n_dim)
    #print("gemm_Test_alpha: ",gmm_test.alpha.shape)
    EM(gmm_test)    