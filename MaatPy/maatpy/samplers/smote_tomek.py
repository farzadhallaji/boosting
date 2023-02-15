from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

__all__ = ['SMOTETomek']


class SMOTETomek(SMOTETomek):
    """
    Class to perform over-sampling using SMOTE and cleaning using Tomek links.
    Combine over- and under-sampling using SMOTE and Tomek links.

    Inherits from imblearn.combine.SMOTETomek. Edited to perform under-sampling first to remove Tomek links
    and then perform oversampling with SMOTE.
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 smote=None,
                 tomek=None):
        """

        :param ratio: str, dict, or callable, optional (default='auto')
               Ratio to use for resampling the data set.
               - If "str", has to be one of: (i) 'minority': resample the minority class;
                 (ii) 'majority': resample the majority class,
                 (iii) 'not minority': resample all classes apart of the minority class,
                 (iv) 'all': resample all classes, and (v) 'auto': correspond to 'all' with for over-sampling
                 methods and 'not_minority' for under-sampling methods. The classes targeted will be over-sampled or
                 under-sampled to achieve an equal number of sample with the majority or minority class.
               - If "dict`", the keys correspond to the targeted classes. The values correspond to the desired number
                 of samples.
               - If callable, function taking "y" and returns a "dict". The keys correspond to the targeted classes.
                 The values correspond to the desired number of samples.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :param smote: object, optional (default=SMOTE())
               The :class: imblearn.over_sampling.SMOTE object to use. If none provide a
               :class: imblearn.over_sampling.SMOTE object with default parameters will be given.
        :param tomek: object, optional (default=TomekLinks())
               The :class: imblearn.under_sampling.TomekLinks object to use. If none provide a
               :class: imblearn.under_sampling.TomekLinks object with default parameters will be given.
        """
        super(SMOTETomek, self).__init__(ratio=ratio,
                                         random_state=random_state,
                                         smote=smote,
                                         tomek=tomek)

    def fit(self, X, y):
        """
        Find the classes statistics before to perform sampling.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: object; Return self

        """
        return super(SMOTETomek, self).fit(X, y)

    def _validate_estimator(self):
        """
        Private function to validate SMOTE and ENN objects
        :return:
        """

        if self.smote is not None:
            if isinstance(self.smote, SMOTE):
                self.smote_ = self.smote
            else:
                raise ValueError('smote needs to be a SMOTE object.'
                                 'Got {} instead.'.format(type(self.smote)))
        else:
            self.smote_ = SMOTE(ratio=self.ratio, k_neighbors=3,
                                random_state=self.random_state)

        if self.tomek is not None:
            if isinstance(self.tomek, TomekLinks):
                self.tomek_ = self.tomek
            else:
                raise ValueError('tomek needs to be a TomekLinks object.'
                                 'Got {} instead.'.format(type(self.tomek)))
        else:
            self.tomek_ = TomekLinks(ratio="all",
                                     random_state=self.random_state)

    def _sample(self, X, y):
        """
        Edited to apply TomekLinks first to remove problematic samples and then apply SMOTE.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the data which have to be sampled.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: X_resampled, y_resampled
        """
        self._validate_estimator()
        X_res, y_res = self.tomek_.fit_sample(X, y)

        return self.smote_.fit_sample(X_res, y_res)
