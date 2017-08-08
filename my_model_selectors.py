import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from operator import itemgetter

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError
    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        def calc_BIC(num_state):
            try:
                model=self.base_model(num_state)
                logL=model.score(self.X,self.lengths)
                p=num_state**2+2*len(self.X[0])*num_state-1
                logN=np.log(len(self.X))
                bic=-2*logL+ p*logN
                return bic
            except:
                return float('-inf')

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        component_bic_tuple=[(i,calc_BIC(i)) for i in range(self.min_n_components,self.max_n_components+1)]
        best_num_components=max(component_bic_tuple,key=itemgetter(1))[0]      
        return self.base_model(best_num_components)
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        def calc_DIC(num_state):
            try:
                model=self.base_model(num_state)
                M=len(self.words)
                scorei=model.score(self.X,self.lengths)
                other_words=list(self.words)
                other_words.remove(self.this_word)
                other_scores=sum([model.score(*self.hwords[w]) for w in other_words])
                dic_score= scorei - other_scores/(M-1)
                return dic_score
            except:
                return float('-inf')

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        component_dic_tuple=[(i,calc_DIC(i)) for i in range(self.min_n_components,self.max_n_components+1)]
        best_num_components=max(component_dic_tuple,key=itemgetter(1))[0]      
        return self.base_model(best_num_components)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        def calc_log(num_state,X_train,length_train,X_test,length_test):
            try:
                hmm_model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train,length_train)
                logL=hmm_model.score(X_test,length_test)
                return logL
            except:
                return float('-inf')        
        def calc_CV_log(num_state,sequences):
            split_method = KFold(2)
            log_value=[]
            if len(sequences)>1:
                for cv_train_idx, cv_test_idx in split_method.split(sequences):
                    X_train,length_train=combine_sequences(cv_train_idx,sequences)
                    X_test,length_test=combine_sequences(cv_test_idx,sequences)
                    log_value.append(calc_log(num_state,X_train,length_train,X_test,length_test))
                return sum(log_value)/len(log_value)
            else:
                return float('-inf')

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        component_log_tuple=[(i,calc_CV_log(i,self.sequences)) for i in range(self.min_n_components,self.max_n_components+1)]
        
        best_num_components=max(component_log_tuple,key=itemgetter(1))[0]
        return self.base_model(best_num_components)

