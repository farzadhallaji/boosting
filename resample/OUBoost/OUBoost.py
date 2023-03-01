import warnings
import matplotlib.pyplot as plt
try:
    get_ipython().magic("matplotlib inline")
except:
    plt.ion()
warnings.simplefilter('ignore', FutureWarning)
from collections import Counter
import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifierMe ## my code saved in C:\Users\Sahar rayaneh\AppData\Local\Programs\Python\Python38\Lib\site-packages\sklearn\ensemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.preprocessing import normalize
from sklearn.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.neighbors import NearestNeighbors
import pandas as pd
#import _weight_boosting



class Sampler(object):
    """Implementation of random undersampling (RUS).
    Undersample the majority class(es) by randomly picking samples with or
    without replacement.
    Parameters
    ----------
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        from the majority class.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self, k_neighbors=5, with_replacement=False, return_indices=False,
                 random_state=None):
        self.k = k_neighbors
        self.return_indices = return_indices
        self.with_replacement = with_replacement
        self.random_state = random_state

    def sampleMaj(self, n_samples):
        

        X_train = self.X
        y_train = self.y

        y_train = pd.DataFrame(y_train)[0]
       # self.y_syn = pd.DataFrame(self.y_syn)[0]



        X_train = np.ascontiguousarray(X_train,dtype=np.float64)
    

        X_train_majority = X_train[y_train==0]
        X_train_minority = X_train[y_train==1]
       # X_train_minority = self.X_syn

        y_train_majority = y_train[y_train==0]
        y_train_minority = y_train[y_train==1]
        #y_train_minority = self.y_syn


        y_train_minority = y_train_minority.reset_index(drop=True)
        y_train_majority = y_train_majority.reset_index(drop=True)



        import pydpc
        from pydpc import Cluster
        #dpc = Cluster(X_train_majority,fraction=0.01,autoplot=False)
        import clustering_selection

        cluster_index,clusters_density,cluster_distance,cluster_ins_den = clustering_selection.clustering_dpc(
              X_train_majority,X_train_minority,y_train_majority,y_train_minority,0,0)

        alpha = 0.5
        beta = 0.5

        X_train_balanced, y_train_balanced, indexs = clustering_selection.selection(X_train_majority, X_train_minority, y_train_majority,
                                                                    y_train_minority, cluster_index, clusters_density,
                                                                    cluster_distance, alpha, beta, cluster_ins_den)
              
        #print("X_train_balanced",X_train_balanced.shape)
       # for i in indexs:
        #         print(i,y_train_majority[i])
        return indexs


    def fitMaj(self, X_org,y_org):

        self.X = X_org
        self.y = y_org
        
        self.n_majority_samples, self.n_features = self.X.shape

        return self
    
    def sampleMin(self, n_samples):
        """Generate samples.
        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.
        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)
        

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1),
                                       return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]
        
        return S

    def fitMin(self, X_min):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X_min
 
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self


class OUBoost(AdaBoostClassifierMe):
    """Implementation of RUSBoost.
    RUSBoost introduces data sampling into the AdaBoost algorithm by
    undersampling the majority class using random undersampling (with or
    without replacement) on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    min_ratio : float (default=1.0)
        Minimum ratio of majority to minority class samples to generate.
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] C. Seiffert, T. M. Khoshgoftaar, J. V. Hulse, and A. Napolitano.
           "RUSBoost: Improving Classification Performance when Training Data
           is Skewed". International Conference on Pattern Recognition
           (ICPR), 2008.
    """

    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=False,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.min_ratio = min_ratio
        self.algorithm = algorithm
        self.ou = Sampler(with_replacement=with_replacement,
                                      return_indices=True,
                                      random_state=random_state)

        super(OUBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            X_org = X
            y_org = y
            sample_weight_org = sample_weight
            
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        OX_min = X_org[np.where(y_org == self.minority_target)]


        for iboost in range(self.n_estimators):
           # print("iboost", iboost)
          #  print("orgx",X_org.shape)
           # print("orgy",y_org.shape)
          #  print("orgw",sample_weight_org.shape)


            # Random undersampling step.
            X_maj = X_org[np.where(y_org != self.minority_target)]
            X_min = X_org[np.where(y_org == self.minority_target)]
            
            stats_ = Counter(y_org == 1)
           # print(stats_)

            ratio=np.sum(y_org)/ y_org.shape[0]
     
          #  print(ratio)
            if  ratio <0.50:
            

             self.ou.fitMin(X_min)
             X_syn = self.ou.sampleMin(self.n_samples)
             y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                              dtype=np.int64)
              # Normalize synthetic sample weights based on current training set.
             sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
             sample_weight_syn[:] = 1. / (X_org.shape[0])
        #     print("syn",sample_weight_syn)

           #  np.random.seed(seed=self.random_state)

            # if X_min.shape[0] > self.n_samples:
        
           #  idx1 = np.random.choice(X_syn.shape[0], size = self.n_samples,replace=False)
             
               
            # X_min_s = X_syn[np.where(y_syn == self.minority_target)][idx1]
             #y_min_s = y_syn[np.where(y_syn == self.minority_target)][idx1]
            # sample_weight_min_s = sample_weight_syn[np.where(y_syn == self.minority_target)][idx1]

             X_org = np.vstack((X_org, X_syn))
             y_org = np.append(y_org, y_syn)

               # Combine the weights.
             sample_weight_org = \
                 np.append(sample_weight_org, sample_weight_syn).reshape(-1, 1)
             sample_weight_org = \
                 np.squeeze(normalize(sample_weight_org, axis=0, norm='l1'))
 
              # Combine the original and synthetic samples.
            # X_org = np.vstack((X_org, X_syn))
             #y_org = np.append(y_org, y_syn)

             # Combine the weights.
         #    sample_weight_org = \
          #       np.append(sample_weight_org, sample_weight_syn).reshape(-1, 1)
           #  sample_weight_org = \
            #     np.squeeze(normalize(sample_weight_org, axis=0, norm='l1'))
             #print("orgx",X_org.shape)
            # print("orgy",y_org.shape)
             #print("orgw",sample_weight_org.shape)
             


             self.ou.fitMaj(X_org,y_org)
            
           #  n_maj = X_maj.shape[0]
           #  n_min = X_min.shape[0]
            
            # if n_maj - self.n_samples < int(n_min * self.min_ratio):
             #    self.n_samples = n_maj - int(n_min * self.min_ratio)
                
             indexs = self.ou.sampleMaj(self.n_samples)
             
             #print(indexs.shape)
            # indexs = np.unique(indexs)
            # print("indexs",indexs.shape)
             
             X_maj = X_org[np.where(y_org != self.minority_target)]
             y_maj = y_org[np.where(y_org != self.minority_target)]
             w_maj = sample_weight_org[np.where(y_org != self.minority_target)]
             

             #print("majx",X_maj.shape)
             #print("majy",y_maj.shape)
             #print("majw",w_maj.shape)


           #  X_rus = X_maj[np.where(y_maj != self.minority_target)][indexs]
             X_rus = np.copy(X_maj)[np.where(y_maj != self.minority_target)][indexs]
            
             X_min = X_org[np.where(y_org == self.minority_target)]
       
  
           #  y_rus = y_maj[np.where(y_maj != self.minority_target)][indexs]
             y_rus = np.copy(y_maj)[np.where(y_maj != self.minority_target)][indexs]
           #  print("majy",y_rus.shape)

             y_min = y_org[np.where(y_org == self.minority_target)]

            # sample_weight_rus = \
             #    w_maj[np.where(y_maj != self.minority_target)][indexs]
             sample_weight_rus = np.copy(w_maj)[np.where(y_maj != self.minority_target)][indexs]
           #  print("maj",sample_weight_rus)

            # print("majw",sample_weight_rus.shape)

             sample_weight_min = sample_weight_org[np.where(y_org == self.minority_target)]
             

 
                    
             X = np.vstack((X_rus, X_min))
             y = np.append(y_rus, y_min)
           

 
              # Combine the weights.
             sample_weight = \
                  np.append(sample_weight_rus, sample_weight_min).reshape(-1, 1)
             sample_weight = \
                  np.squeeze(normalize(sample_weight, axis=0, norm='l1'))
             
 
            # Boosting step.
            sample_weight, estimator_weight, estimator_error,sample_weight_org = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state,X_org,y_org,sample_weight_org)
            # print("error",estimator_error)
             
          
            X = X_org
            y = y_org
            sample_weight = sample_weight_org
            



            # Early termination.
            if sample_weight_org is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight_org)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight_org /= sample_weight_sum


               
        return self

class SMOTE(object):
    """Implementation of Synthetic Minority Over-Sampling Technique (SMOTE).
    SMOTE performs oversampling of the minority class by picking target 
    minority class samples and their nearest minority class neighbors and 
    generating new samples that linearly combine features of each target 
    sample with features of its selected minority class neighbors [1].
    Parameters
    ----------
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE:
           Synthetic Minority Over-Sampling Technique." Journal of Artificial
           Intelligence Research (JAIR), 2002.
    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.
        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.
        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1),
                                       return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

    def fit(self, X):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self


class SMOTEBoost(AdaBoostClassifier):
    """Implementation of SMOTEBoost.
    SMOTEBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class using SMOTE on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
           "SMOTEBoost: Improving Prediction of the Minority Class in
           Boosting." European Conference on Principles of Data Mining and
           Knowledge Discovery (PKDD), 2003.
    """

    def __init__(self,
                 n_samples=100,
                 k_neighbors=5,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.smote = SMOTE(k_neighbors=k_neighbors,
                           random_state=random_state)

        super(SMOTEBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing SMOTE during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            
            # SMOTE step.
            X_min = X[np.where(y == self.minority_target)]
            self.smote.fit(X_min)
            X_syn = self.smote.sample(self.n_samples)
            y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                            dtype=np.int64)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the original and synthetic samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self
class RandomUnderSampler(object):
    """Implementation of random undersampling (RUS).
    Undersample the majority class(es) by randomly picking samples with or
    without replacement.
    Parameters
    ----------
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        from the majority class.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self, with_replacement=True, return_indices=False,
                 random_state=None):
        self.return_indices = return_indices
        self.with_replacement = with_replacement
        self.random_state = random_state

    def sample(self, n_samples):
        """Perform undersampling.
        Parameters
        ----------
        n_samples : int
            Number of samples to remove.
        Returns
        -------
        S : array, shape = [n_majority_samples - n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        if self.n_majority_samples <= n_samples:
            n_samples = self.n_majority_samples

        idx = np.random.choice(self.n_majority_samples,
                               size=self.n_majority_samples - n_samples,
                               replace=self.with_replacement)

        if self.return_indices:
            return (self.X[idx], idx)
        else:
            return self.X[idx]

    def fit(self, X):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_majority_samples, n_features]
            Holds the majority samples.
        """
        self.X = X
        self.n_majority_samples, self.n_features = self.X.shape

        return self


class RUSBooost(AdaBoostClassifier):
    """Implementation of RUSBoost.
    RUSBoost introduces data sampling into the AdaBoost algorithm by
    undersampling the majority class using random undersampling (with or
    without replacement) on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    min_ratio : float (default=1.0)
        Minimum ratio of majority to minority class samples to generate.
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] C. Seiffert, T. M. Khoshgoftaar, J. V. Hulse, and A. Napolitano.
           "RUSBoost: Improving Classification Performance when Training Data
           is Skewed". International Conference on Pattern Recognition
           (ICPR), 2008.
    """

    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.min_ratio = min_ratio
        self.algorithm = algorithm
        self.rus = RandomUnderSampler(with_replacement=with_replacement,
                                      return_indices=True,
                                      random_state=random_state)

        super(RUSBooost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Random undersampling step.
            X_maj = X[np.where(y != self.minority_target)]
            X_min = X[np.where(y == self.minority_target)]
            
            self.rus.fit(X_maj)

            n_maj = X_maj.shape[0]
            n_min = X_min.shape[0]
            if n_maj - self.n_samples < int(n_min * self.min_ratio):
                self.n_samples = n_maj - int(n_min * self.min_ratio)
            X_rus, X_idx = self.rus.sample(self.n_samples)

            y_rus = y[np.where(y != self.minority_target)][X_idx]
            y_min = y[np.where(y == self.minority_target)]

            sample_weight_rus = \
                sample_weight[np.where(y != self.minority_target)][X_idx]
          
            sample_weight_min = \
                sample_weight[np.where(y == self.minority_target)]
           

            # Combine the minority and majority class samples.
          
            X = np.vstack((X_rus, X_min))
            y = np.append(y_rus, y_min)
       
            #print(sum(y))

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight_rus, sample_weight_min).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)
         


            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)
           



            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

               
        return self

from pandas import read_csv
  
#dataframe=read_csv('mammography.csv')
dataframe=read_csv('ecoli_4.csv')

#dataframe=read_csv('D:\\python tests\\imbalanced tests\\rusboost\\mammography 40000\\datasetme.csv')
dim=dataframe.shape
array = dataframe.values    
Y = array[:,dim[1]-1]
X = array[:,0:dim[1]-1]
from sklearn.preprocessing import Normalizer,MinMaxScaler,StandardScaler

from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import os
from pycm import *
import statistics
from scipy.stats.mstats import gmean
from random import randint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=None, shuffle=True)

auc_result_ouboost=[]
auc_result_rus=[]
auc_result_smote=[]
auc_result_ada=[]

gmean_ouboost=[]
gmean_smote=[]
gmean_ada=[]
gmean_rus=[]




score_array_ada=[]
score_array_rus=[]
score_array_smote=[]
score_array_ouboost=[]

result_ouboost=[]
result_smote=[]
result_ada=[]
result_rus=[]

mcc_ada=[]
mcc_rus=[]
mcc_smote=[]
mcc_ouboost=[]
from sklearn.metrics import precision_recall_fscore_support
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
for train_index, test_index in kf.split(X):
     
      X_train, X_test = X[train_index], X[test_index] 
      Y_train, Y_test = Y[train_index], Y[test_index]
      scl = StandardScaler()
      X_train = scl.fit_transform(X_train)
      X_test = scl.transform(X_test)
      X_train=np.ascontiguousarray(X_train,dtype=np.float64)
      X_test=np.ascontiguousarray(X_test,dtype=np.float64)
      #**********************************************
      from datetime import datetime
 
      start = datetime.now()
      model_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=50,algorithm="SAMME", learning_rate=0.3)
      model_ada.fit(X_train, Y_train)
      predictions_ada = model_ada.predict(X_test)
      end = datetime.now()
      td = (end - start).total_seconds() * 10**3
      print(f"The time of execution of Adaboost is : {td:.03f}ms")
      proba_ada = model_ada.predict_proba(X_test)
      score_array_ada.append(precision_recall_fscore_support(Y_test, predictions_ada, average=None))
      result_ada.append(accuracy_score(Y_test, predictions_ada))
      auc_ada = proba_ada[:,1]
      fpr, tpr, threshold = metrics.roc_curve(Y_test, auc_ada)
      auc_result_ada.append(metrics.auc(fpr, tpr))
      gmean_ada.append(geometric_mean_score(Y_test, predictions_ada, average=None))
      mcc_ada.append(matthews_corrcoef(Y_test, predictions_ada))
      print("AdaBoost",accuracy_score(Y_test, predictions_ada))
      print(confusion_matrix(Y_test, predictions_ada))
      print(classification_report(Y_test, predictions_ada))
      print("************************************************************************")

      #*********************
      start = datetime.now()

      classification_smote = SMOTEBoost(learning_rate=0.3, n_samples=10, n_estimators=50)
      classification_smote.fit(X_train, Y_train)
      y_pred_smote = classification_smote.predict(X_test)
      end = datetime.now()
      td = (end - start).total_seconds() * 10**3
      print(f"The time of execution of SMOTEBoost is : {td:.03f}ms")
      proba_smote = classification_smote.predict_proba(X_test)
      auc_smote = proba_smote[:,1]
      fpr, tpr, threshold = metrics.roc_curve(Y_test, auc_smote)
      auc_result_smote.append(metrics.auc(fpr, tpr))
      gmean_smote.append(geometric_mean_score(Y_test, y_pred_smote, average=None))
      score_array_smote.append(precision_recall_fscore_support(Y_test, y_pred_smote, average=None))
      result_smote.append(accuracy_score(Y_test, y_pred_smote))
      mcc_smote.append(matthews_corrcoef(Y_test, y_pred_smote))

      print("SmoteBoost",accuracy_score(Y_test, y_pred_smote))
      print(confusion_matrix(Y_test, y_pred_smote))
      print(classification_report(Y_test, y_pred_smote))

      print("************************************************************************")

      #*******************
      start = datetime.now()
      classification_rusboost = RUSBooost(learning_rate=0.3, n_samples=5, n_estimators=10)
      classification_rusboost.fit(X_train, Y_train)
      y_pred_rus = classification_rusboost.predict(X_test)
      end = datetime.now()
      td = (end - start).total_seconds() * 10**3
      print(f"The time of execution of RUSBoost is : {td:.03f}ms")
      proba_rus = classification_rusboost.predict_proba(X_test)
      auc_rus = proba_rus[:,1]
      fpr, tpr, threshold = metrics.roc_curve(Y_test, auc_rus)
      auc_result_rus.append(metrics.auc(fpr, tpr))
      gmean_rus.append(geometric_mean_score(Y_test, y_pred_rus, average=None))
      score_array_rus.append(precision_recall_fscore_support(Y_test, y_pred_rus, average=None))
      result_rus.append(accuracy_score(Y_test, y_pred_rus))
      mcc_rus.append(matthews_corrcoef(Y_test, y_pred_rus))

      print("RUSBoost",accuracy_score(Y_test, y_pred_rus))
      print(confusion_matrix(Y_test, y_pred_rus))
      print(classification_report(Y_test, y_pred_rus))

      #******************
      start = datetime.now()
      classification_ouboost = OUBoost(learning_rate=0.3, n_samples=100, n_estimators=50)
      classification_ouboost.fit(X_train, Y_train) 
      y_pred_ouboost = classification_ouboost.predict(X_test)
      end = datetime.now()
      td = (end - start).total_seconds() * 10**3
      print(f"The time of execution of OUBoost is : {td:.03f}ms")
      proba_ouboost = classification_ouboost.predict_proba(X_test)
      auc_ouboost = proba_ouboost[:,1]
      fpr, tpr, threshold = metrics.roc_curve(Y_test, auc_ouboost)
      auc_result_ouboost.append(metrics.auc(fpr, tpr))
      gmean_ouboost.append(geometric_mean_score(Y_test, y_pred_ouboost, average=None))
      mcc_ouboost.append(matthews_corrcoef(Y_test, y_pred_ouboost))

      print("OUBoost",accuracy_score(Y_test, y_pred_ouboost))
      print(confusion_matrix(Y_test, y_pred_ouboost))
      print(classification_report(Y_test, y_pred_ouboost))
      score_array_ouboost.append(precision_recall_fscore_support(Y_test, y_pred_ouboost, average=None))
      result_ouboost.append(accuracy_score(Y_test, y_pred_ouboost))
      print("************************************************************************")
      print("************************************************************************")
      print("************************************************************************")
      
#***************************************************
print("AdaBoost Acc:",np.array(result_ada).mean())
avg_score_ada = np.mean(score_array_ada,axis=0)
print(avg_score_ada)
print("AdaBoost Auc: ", np.array(auc_result_ada).mean())
print("AdaBoost Gmean: ", np.array(gmean_ada).mean())
print("AdaBoost MCC:",np.array(mcc_ada).mean())

print("************************************************************************")
#**************************************************
print("RusBoost Acc:",np.array(result_rus).mean())
avg_score_rus = np.mean(score_array_rus,axis=0)
print(avg_score_rus)
print("RusBoost Auc: ", np.array(auc_result_rus).mean())
print("RusBoost Gmean: ", np.array(gmean_rus).mean())
print("RusBoost MCC:",np.array(mcc_rus).mean())
print("************************************************************************")

#*******************************************
print("SmoteBoost Acc:",np.array(result_smote).mean())
avg_score_smote = np.mean(score_array_smote,axis=0)
print(avg_score_smote)
print("SmoteBoost Auc: ", np.array(auc_result_smote).mean())
print("SmoteBoost Gmean: ", np.array(gmean_smote).mean())
print("SmoteBoost MCC:",np.array(mcc_smote).mean())

print("************************************************************************")
#********************************************
print("OUBoost Acc:",np.array(result_ouboost).mean())
avg_score_ouboost = np.mean(score_array_ouboost,axis=0)
print(avg_score_ouboost)
print("OUBoost Auc: ", np.array(auc_result_ouboost).mean())
print("OUBoost Gmean: ", np.array(gmean_ouboost).mean())
print("OUBoost MCC:",np.array(mcc_ouboost).mean())

            

