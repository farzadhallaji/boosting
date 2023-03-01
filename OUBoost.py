from collections import Counter
import numpy as np 
import pandas as pd
import pydpc
from pydpc import Cluster

from sklearn.ensemble import AdaBoostClassifierMe

from sklearn.utils import check_X_y
from sklearn.base import is_regressor
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


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

        X_train = np.ascontiguousarray(X_train,dtype=np.float64)
        
        y_uniques, y_counts = np.unique(y_train, return_counts=True)
        maj_y = y_uniques[np.argmax(y_counts)]
        mini_y = y_uniques[np.argmin(y_counts)]
        
        # print(X_train[y_train==maj_y])
        X_train_majority = X_train[y_train==maj_y]
        X_train_minority = X_train[y_train==mini_y]

        y_train_majority = y_train[y_train==maj_y]
        y_train_minority = y_train[y_train==mini_y]


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

        self._validate_estimator()


        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        OX_min = X_org[np.where(y_org == self.minority_target)]


        for iboost in range(self.n_estimators):

            # Random undersampling step.
            X_maj = X_org[np.where(y_org != self.minority_target)]
            X_min = X_org[np.where(y_org == self.minority_target)]
            
            stats_ = Counter(y_org == 1)

            ratio=np.sum(y_org)/ y_org.shape[0]
     
            if  ratio <0.50:
                self.ou.fitMin(X_min)
                X_syn = self.ou.sampleMin(self.n_samples)
                y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                              dtype=np.int64)
                # Normalize synthetic sample weights based on current training set.
                sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
                sample_weight_syn[:] = 1. / (X_org.shape[0])

                X_org = np.vstack((X_org, X_syn))
                y_org = np.append(y_org, y_syn)

                # Combine the weights.
                sample_weight_org = \
                 np.append(sample_weight_org, sample_weight_syn).reshape(-1, 1)
                sample_weight_org = \
                 np.squeeze(normalize(sample_weight_org, axis=0, norm='l1'))
                self.ou.fitMaj(X_org,y_org)
                
                indexs = self.ou.sampleMaj(self.n_samples)
                X_maj = X_org[np.where(y_org != self.minority_target)]
                y_maj = y_org[np.where(y_org != self.minority_target)]
                w_maj = sample_weight_org[np.where(y_org != self.minority_target)]
                #  X_rus = X_maj[np.where(y_maj != self.minority_target)][indexs]
                
                print('indexs.max(), X_maj.shape', indexs.max(), X_maj.shape)
                X_rus = np.copy(X_maj)[np.where(y_maj != self.minority_target)][indexs]
                
                
                
                X_min = X_org[np.where(y_org == self.minority_target)]
                #  y_rus = y_maj[np.where(y_maj != self.minority_target)][indexs]
                y_rus = np.copy(y_maj)[np.where(y_maj != self.minority_target)][indexs]
                y_min = y_org[np.where(y_org == self.minority_target)]
                sample_weight_rus = np.copy(w_maj)[np.where(y_maj != self.minority_target)][indexs]
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
