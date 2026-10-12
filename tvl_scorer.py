# In this file we:
# -compute the TvL measure (compute_tvl)
# -perform cross validation with it (perform_cross_validation_tvl)
# -perform cross validation with accuracy (perform_cross_validation_accuracy)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from matplotlib.pylab import seed
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class TalentWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None):
        # BaseEstimator requires parameters to be assigned directly to self
        self.estimator = estimator

    def fit(self, X, y):
        # 1. Clone to ensure a new model for every fold
        self.estimator_ = clone(self.estimator)
        
        # 2. Fit on the training fold (X and y here are ONLY the training fold)
        self.estimator_.fit(X, y)
        
        # 3. Calculate 'talent' while we have the training data
        y_train_pred = self.estimator_.predict(X)
        self.talent_ = accuracy_score(y, y_train_pred)
        
        # Required to return self
        return self

    def predict(self, X):
        # Passes the validation data to the internal fitted model
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        # useful for AUC or log-loss 
        return self.estimator_.predict_proba(X)

def tvl_scorer(estimator, X, y, m, L):
    '''
    Inputs:
        estimator: fitted estimator
        X: vector of floats, validation features which refers to the validation set when TvL is used for HP
        y: vector of floats, true labels which refers to the validation set when TvL is used for HP
        m: int, cardinality of each subset of y
        L: int, number of subsets
    Outputs:
        TvL: float, TvL measure
    '''
    # Retrieve pre-calculated talent from the training fold
    talent = estimator.talent_
    y_pred = estimator.predict(X)

    if len(y) < m:
        print("Error: m cannot be larger than the size of testSet.")
        return None    
    
    c = 0 # Initialize counter
    rng = np.random.default_rng(42)
    
    for _ in range(L):
        indices = rng.choice(len(y), size=m, replace=False)
        subset_y = y[indices]
        subset_pred = y_pred[indices]
        
        if accuracy_score(subset_y, subset_pred) >= talent:
            c += np.sum(subset_y == subset_pred)
    tvl = c / (L * m)
    return tvl

def get_measures(estimator, X_train, X_test, y_train, y_test):
    
    # Assumes that estimator has already been fitted on the training data. 

    # Get predictions
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    
    # Calculate accuracy
    train_acc = (accuracy_score(y_train, y_train_pred)) * 100
    test_acc = (accuracy_score(y_test, y_test_pred)) * 100


    # Calculate Generalization Error
    gen_error = train_acc - test_acc

    # Classification report
    clf_report = classification_report(y_test, y_test_pred)

    # Calculate AUROC (handle both binary and multi-class)
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        # Binary classification
        y_probs = estimator.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_probs)
    else:
        # Multi-class classification
        y_probs = estimator.predict_proba(X_test)
        auroc = roc_auc_score(y_test, y_probs, multi_class='ovr')

    measures = {
        'training accuracy': train_acc,
        'test accuracy': test_acc,
        'generalization error': gen_error,
        'AUROC': auroc,
        'classification report': clf_report
    }

    for measure, value in measures.items():
        if measure == 'classification report':
            print(f"{measure}:\n{value}")
        else:
            print(f"{measure}: {value:.4f}")
    return measures

# Helper function to convert numpy types to Python native types
def convert_to_python_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj