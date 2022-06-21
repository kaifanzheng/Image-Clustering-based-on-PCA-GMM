import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import operator

#--------------------------------------------------assignment two code start
from cmath import pi
from nntplib import NNTP
from telnetlib import GA
from turtle import color, shape
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from time import sleep
import time

class GMM:
    def __init__( self, X, n_components = 10, reg_covar = 1e-2, tol = 1e-4, 
                  max_iter = 100, verbose = True, do_plot = False, mu_init = None):

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
            plt.pause(0.1)
            sleep(0.1)
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
def normalization(gmm:GMM):     
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
mus = []
pis = []
covs = []
def maximization(gmm:GMM):                   
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
        sum = 0
        for i in range(gmm.n_samples):
            sum = sum+(gmm.alpha[i][j]*gmm.X[i])
        gmm.mu[j] = (1/gmm.weight[j])*sum
    
    for j in range(gmm.n_components):
        sum = 0
        for i in range(gmm.n_samples):
            xsu = (gmm.X[i] - gmm.mu[j])[np.newaxis]
            transXsu =  np.transpose((gmm.X[i] - gmm.mu[j])[np.newaxis])
            sum = sum+ (gmm.alpha[i][j]*(transXsu@xsu))
        gmm.cov[j] = (sum+gmm.reg_cov)*(1/gmm.weight[j])

    # record mu cov and pi for each interation
    # mus.append(gmm.mu)
    # covs.append(gmm.cov)
    # pis.append(gmm.pi)

    # print("mu:  ")
    # print(gmm.mu)
    # print("cov: ")
    # print(gmm.cov)
    # print("beta: ")
    # print(gmm.beta)
    # print("alpha: ")
    # print(gmm.alpha)
    # print("pi")
    # print(gmm.pi)
    # print(" " )

    Gaussians = []
    for i in range(gmm.n_components):
        Gaussians.append([multivariate_normal( mean = gmm.mu[i] , cov = gmm.cov[i])])
    gmm.gauss = Gaussians
    ### END SOLUTION


# [TODO] Deliverable 7: Compute the log-likelihood
def logLikelihood(gmm:GMM):                        

    # Note: you need to append to gmm.log_likelihoods
    
    ### BEGIN SOLUTION
    #print("betas: ",gmm.beta)
    # sum = 0
    # for i in range(gmm.n_samples):
    #     sum = sum+np.log(gmm.beta[i])
    #print("beta: ",gmm.beta)
    sum = (np.log(gmm.beta)).sum()
    gmm.log_likelihoods.append(sum)
    #print(sum)
    #print("likelyhood: ",gmm.log_likelihoods)
    ### END SOLUTION
#--------------------------------------------------assignment two code end
colorLabel = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128],[0,128,128],[192,192,192],[128,128,128],[153,153,153],[153,51,102],[255,255,204],[204,255,255],[102,0,102],[255,128,128],[0,102,204],[204,204,255],[0,0,80],[51,51,0],[153,51,0],[152,51,0],[153,51,102],[51,51,51],[51,51,153],[104,135,255]]).astype(int)
colorLabelGray = np.array([12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252]).astype(int)
#pca for Dimensionality Reduction, reduce RGB into 2 demension
def PCA(points: np.array, n_dimention:int):
    #find Mean value for X _T by 0
    mean = points - np.mean(points,axis=0)
    covariant = np.cov(mean,rowvar=False)#calculate covariance matrix
    values, vectors = np.linalg.eigh(covariant)#calculate eigenvalue and eigenvector
    #ref: https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
    #sort values and vectors in accending order
    idx = np.argsort(values)[::-1]
    values = values[idx]#sorted eigen values
    vectors = vectors[:,idx]#sorted eigen vectors
    vectors_n_dimention  = vectors[:,0:n_dimention]#reduce input to n dimention
    V = np.dot(vectors_n_dimention.T ,mean.T)
    return (V.T,values) #get the eigen reduced dimension
#reduce a demension of the img, in order to process PCA
def flatImg(img:np.array):
    result = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result.append(img[i][j])
    return np.array(result)
#rebuild the image and convert back to 3d
def rebuildImg(flatImg:np.array,shape,n_dimonsion):
    result = np.zeros((shape[0],shape[1],n_dimonsion))
    counter = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i][j] = flatImg[counter]
            counter +=1
    return result

def pyrimid_img(img:np.array,pyrimidSize:int):
    def resizeImg(img:np.array, scale_percentage:int):
        width = int(img.shape[1] * (scale_percentage/100))
        height = int(img.shape[0] * (scale_percentage/100))
        theShape = (width, height)
        return cv.resize(img,theShape)
    resultImgs = []
    precent = []
    for i in range(1,pyrimidSize):
        precent.append(int(i*(100/pyrimidSize)))
    precent.append(100)
    # for pre in precent:
    #     resultImgs.append(resizeImg(img,pre))
    resultImgs.append(resizeImg(img,precent[0]))
    resultImgs.append(resizeImg(img,precent[-1]))
    return resultImgs
def resizeParameter(parameter,oldShape,newShape):
    reshapePara = np.array(parameter.reshape(oldShape),dtype='uint8')
    return flatImg(cv.resize(reshapePara,newShape))

def convertToGrayMean(img:np.array):
    result = np.mean(img,axis=2)
    return result.astype(int)
#return both labeled image and average colored image
def optimizedImgSeg(img:np.array, pyrimidSize:int,k:int,do_plot:bool):
    Imgs = pyrimid_img(img,pyrimidSize)
    eigenVectoers = []
    for img in Imgs:
        imgFlat = flatImg(img)
        eigenVec = PCA(imgFlat,2)[0]
        eigenVec = eigenVec+np.abs(np.min(eigenVec))
        eigenVectoers.append(eigenVec)
    #initilize the first interation parameter
    gmm_test  = GMM( eigenVectoers[0], k, reg_covar = 1e-3, 
                    tol = 1e-6, max_iter = 300, 
                    verbose = True, do_plot = do_plot) 
    testImgShape = Imgs[0].shape
    EM(gmm_test)
    #now we have lables from the em result, scale the result into the original picture
    Z = np.array(gmm_test.Z,dtype=np.uint8)
    labeledImg = np.zeros(testImgShape,dtype=np.uint8)
    counter = 0
    for i in range(testImgShape[0]):
        for j in range(testImgShape[1]):
            labeledImg[i][j] = colorLabelGray[Z[counter]]
            counter += 1
    labeledImg = cv.resize(labeledImg,(Imgs[1].shape[1],Imgs[1].shape[0]))
    cv.imwrite("resultImg/ele_labeled_afterEM.jpeg",labeledImg)#write the image to see the test result
    dic = {}
    points = []
    counter = 0
    for i in range(labeledImg.shape[0]):
        for j in range(labeledImg.shape[1]):
            points.append(K_point(eigenVectoers[1][counter][0],eigenVectoers[1][counter][1],[i,j],labeledImg[i][j]))#j is x, i is y
            if int(labeledImg[i][j][0]) in dic.keys():
                dic[int(labeledImg[i][j][0])].append(eigenVectoers[1][counter])
            else:
                dic[int(labeledImg[i][j][0])] = [eigenVectoers[1][counter]]
            counter += 1
    #print("dictionary keys: ",dic)
    centers = []
    for k in sorted(dic, key=lambda k: len(dic[k]), reverse=True)[0:k]:
        centers.append(np.array(dic[k]).mean(axis=0))
    #print("centers: ",centers)
    newKPoint = kMean_2d(centers,points,100)
    resultImg_labeled = convertPointsToLabledImage(newKPoint,Imgs[1].shape)
    resultImg = convertPointsToImage(newKPoint,Imgs[1])
    return resultImg_labeled.astype(int), resultImg.astype(int)
#points what has attribute of x and y feature and label and cordinate, and color
class K_point:
    def __init__(self,X, Y,cor,label):
        self.X = X
        self.Y =Y
        self.label = label
        self.cor = cor#cordinate of the point, for reconstract image
        self.color = [0,0,0]
    def findDistance(self,x,y):
        return (((x-self.X)**2) + ((y - self.Y)**2))**0.5 

#centers is a list of center, the len(center) is the value of k
def kMean_2d(centers,kpoints:list,n_iter:int):
    prevCenters = None
    for i in range(n_iter):
        print("kMean_2d iteration",i)
        # np.array(sorted(centers,key=lambda x:x[1]))
        prevCenters = centers
        dic = {}#initilize the dictionary to sort the points in the same lable
        for point in kpoints:
            #distance has content ((distance,lable))
            distance = []
            for i in range(len(centers)):
                dis = point.findDistance(centers[i][0],centers[i][1])
                distance.append((dis,i))
            distance = sorted(distance)
            point.label = distance[0][1] #update the lable of the point
            if point.label in dic.keys():
                dic[point.label].append((point.X,point.Y))
            else:
                dic[point.label] = [(point.X,point.Y)]
        # for key in dic.keys():
        #     print(len(dic[key]))
        #the dictionary is ready for new centers
        counter = 0
        for keys in dic.keys():
            centers[counter] = np.array(dic[keys]).mean(axis=0)#new center is (x,y)
        np.array(sorted(centers,key=lambda x:x[1]))
        # print("new center: ",centers)
        # print("previous center, ",prevCenters)
        if np.allclose(prevCenters,centers):#when the center point doesn't chang, it's time
            break
    return kpoints
def convertPointsToLabledImage(kpoints,shape):
    result = np.zeros(shape)
    for points in kpoints:
        result[points.cor[0]][points.cor[1]] = colorLabel[points.label]
    return result
def convertPointsToImage(kpoints,original_img):
    result = np.zeros(original_img.shape)
    dic = {}
    counter = 0
    for point in kpoints:
        if point.label in dic.keys():
            dic[point.label].append(original_img[point.cor[0]][point.cor[1]])
        else:
            dic[point.label] = [original_img[point.cor[0]][point.cor[1]]]
    for key in dic.keys():
        color =  np.array(dic[key]).mean(axis=0)
        for point in kpoints:
            if point.label == key:
                point.color = color
    for point in kpoints:
        result[point.cor[0]][point.cor[1]] = point.color
    return result
    

def showImage(subplot, image, title):
    plt.subplot(subplot), plt.imshow(image)
    plt.title(title), plt.xticks([]), plt.yticks([])
def testSetOne():
    print("testing the logic of segmentatio:----- \n")
    testImg = cv.imread("testImg/ele.jpeg")
    print("the shape of test image is: ",testImg.shape,"\n")
    testImgShape = testImg.shape
    print("task one: PCA testing, reduce to dimension one\n")
    testImgVector = flatImg(testImg)
    eigenVector,Values = PCA(testImgVector,1)
    #Variance ratio
    plt.figure()
    plt.plot(np.cumsum(Values) / np.sum(Values))
    plt.xlabel("Number of PCs")
    plt.ylabel("Explained Variance")
    plt.show()
    #Normalized eigenvalues
    normEigenvalues = Values / np.max(Values)
    plt.figure()
    plt.plot(normEigenvalues)
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Normalized Eigenvalues")
    plt.show()

    plt.imshow(rebuildImg(eigenVector,testImgShape,1))
    plt.title("eigen Image with one PC")
    plt.show()
    print("task two: use PCA to convert RGB value to 2 dimensional plot them in 2D graph\n")
    testImgVector = flatImg(testImg)
    eigenVector,Values = PCA(testImgVector,2)#reduce input testImgVector's RGB into 2 demension
    #print(np.min(eigenVector))
    eigenVector = eigenVector+np.abs(np.min(eigenVector))
    print(eigenVector)
    plt.title("plot 2 PCs")
    plt.scatter(eigenVector[:,0],eigenVector[:,1])
    plt.show()
    print("task three: test clusting use Gaussian mixture models\n")
    gmm_test  = GMM( eigenVector, 4, reg_covar = 1e-3, 
                    tol = 1e-6, max_iter = 200, 
                    verbose = True, do_plot = False) 
    start_time = time.time()
    EM(gmm_test)
    print("the algorithm is too slow, the run time is: ","--- %s seconds ---" % (time.time() - start_time),"\n")
    print("task four: generate the labeled image\n")
    # colorLabel = [0,25,50,75,90,115,140,165,190,215,240]
    labeledImg = np.zeros(testImgShape)
    counter = 0
    for i in range(testImgShape[0]):
        for j in range(testImgShape[1]):
            labeledImg[i][j] = colorLabel[gmm_test.Z[counter]]
            counter += 1
    cv.imwrite("resultImg/bear_labeled.jpeg",labeledImg)
    plt.title("labeled image")
    plt.imshow(labeledImg)
    plt.show()

def testSetTwo():
    print("testing the logic of optimize segmentation:----- \n")
    print("task five: test pyrimid image function\n")
    testImg = cv.imread("testImg/ele.jpeg")
    pyrimideImg = pyrimid_img(testImg,4)
    for img in pyrimideImg:
        print("the shape of image is: ", img.shape)
    print("task six: test the potimized segmentation")
def experimentOneGenerateImg(path):
    #experimentTime = 3
    experimentK = (7,10)
    for j in range(experimentK[0],experimentK[1]+1):
        testImg = cv.imread(path)
        #print(testImg)
        np.random.seed(2)
        testImgShape = testImg.shape
        resultImg_labled,resultImg = optimizedImgSeg(testImg,8,j,False)
        title = "labeled image k: "+str(j)
        titleImg = "average image k: "+str(j)
        cv.imwrite("resultImg/"+title+".jpeg",resultImg_labled)
        cv.imwrite("resultImg/"+titleImg+".jpeg",resultImg)
        plt.figure(figsize=(10,10))
        showImage(121,resultImg_labled,"labeled img")
        showImage(122,resultImg,"average img")
        plt.show()
def experimentComputingSpeed():
    print("speed of segmentation without optimize:----- \n")
    np.random.seed(2)
    start_time = time.time()
    testImg = cv.imread("testImg/ele.jpeg")
    testImgShape = testImg.shape
    testImgVector = flatImg(testImg)
    eigenVector,Values = PCA(testImgVector,1)

    testImgVector = flatImg(testImg)
    eigenVector,Values = PCA(testImgVector,2)#reduce input testImgVector's RGB into 2 demension
    #print(np.min(eigenVector))
    eigenVector = eigenVector+np.abs(np.min(eigenVector))
    gmm_test  = GMM( eigenVector, 3, reg_covar = 1e-3, 
                    tol = 1e-6, max_iter = 200, 
                    verbose = True, do_plot = False) 
    start_time = time.time()
    EM(gmm_test)
    # colorLabel = [0,25,50,75,90,115,140,165,190,215,240]
    labeledImg = np.zeros(testImgShape)
    counter = 0
    for i in range(testImgShape[0]):
        for j in range(testImgShape[1]):
            labeledImg[i][j] = colorLabel[gmm_test.Z[counter]]
            counter += 1
    print("---speed of segmentation without optimize: %s seconds ---" % (time.time() - start_time))
    print("speed of segmentation after optimize:----- \n")
    np.random.seed(2)
    start_time = time.time()
    testImg = cv.imread("testImg/ele.jpeg")
    resultImg_labled,resultImg = optimizedImgSeg(testImg,8,3,False)
    print("---speed of segmentation after optimize: %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    pathOne = "testImg/ele.jpeg"
    pathTwo = "bearTwo.jpeg" #if you want to change the input image, change the path
    #experimentComputingSpeed()
    #testSetOne()#test the pca,gmm class and em function
    #testSetTwo()#test the primid function
    experimentOneGenerateImg(pathOne)
    print("all the result stores in the 'resultImg' folder")