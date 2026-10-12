import numpy as np
import optuna
from functools import partial
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from tvl_scorer import tvl_scorer, TalentWrapper


def grid_search_svm(X, y, iterations=3, initial_c_range=(1e-2, 1e2), initial_gamma_range=(1e-4, 1e0),
                    degrees=None, steps=5, kernel='rbf',
                    scoring='accuracy', m=10, L=100, random_seed=42, n_jobs=-1):
    """
    Performs iterative Coarse-to-Fine grid search for SVM.
    
    Args:
        X, y: Training data.
        iterations: Number of times to narrow the search.
        initial_c_range: Tuple of (start_exponent, end_exponent) for C logspace.
        initial_gamma_range: Tuple of (start_exponent, end_exponent) for gamma logspace.
        degrees: Tuple of (start_degree, end_degree) for polynomial kernel (used only if kernel='poly').
        steps: Number of points to test per axis.
        kernel: The kernel type for the SVM ('rbf' or 'poly').
        scoring: 'accuracy' or 'tvl'.
    """
    curr_c_space     = np.logspace(initial_c_range[0], initial_c_range[1], steps)
    curr_gamma_space = np.logspace(initial_gamma_range[0], initial_gamma_range[1], steps)
    d_space = np.arange(degrees[0], degrees[1] + 1) if degrees is not None else None

    best_params = None
    best_score  = -np.inf
    zoom_factors = [None, 0.5, 0.2]

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    for i in range(iterations):
        print(f"--- Iteration {i+1} ---")
        print(f"Searching C in {curr_c_space}, Gamma in {curr_gamma_space}")

        if scoring == 'accuracy':
            param_grid = ({'C': curr_c_space, 'gamma': curr_gamma_space, 'degree': d_space}
                          if kernel == 'poly' else
                          {'C': curr_c_space, 'gamma': curr_gamma_space})
            print("Accuracy tuning:")
            search = GridSearchCV(SVC(kernel=kernel, probability=True, random_state=random_seed),
                                  param_grid, cv=cv_strategy, scoring='accuracy',
                                  n_jobs=n_jobs, verbose=3)
            search.fit(X, y)
            best_params = search.best_params_
            if i < iterations - 1:
                factor = zoom_factors[i + 1]
                curr_c_space     = np.linspace(best_params['C']     * (1 - factor), best_params['C']     * (1 + factor), steps)
                curr_gamma_space = np.linspace(best_params['gamma'] * (1 - factor), best_params['gamma'] * (1 + factor), steps)

        elif scoring == 'tvl':
            param_grid = ({'estimator__C': curr_c_space, 'estimator__gamma': curr_gamma_space, 'estimator__degree': d_space}
                          if kernel == 'poly' else
                          {'estimator__C': curr_c_space, 'estimator__gamma': curr_gamma_space})
            print("TvL tuning:")
            talent_wrapper = TalentWrapper(
                estimator=SVC(kernel=kernel, probability=True, random_state=random_seed))
            objective_score = partial(tvl_scorer, m=m, L=L)
            search = GridSearchCV(talent_wrapper, param_grid, cv=cv_strategy,
                                  scoring=objective_score, n_jobs=n_jobs, verbose=3)
            search.fit(X, y)
            best_params = search.best_params_
            if i < iterations - 1:
                factor = zoom_factors[i + 1]
                curr_c_space     = np.linspace(best_params['estimator__C']     * (1 - factor), best_params['estimator__C']     * (1 + factor), steps)
                curr_gamma_space = np.linspace(best_params['estimator__gamma'] * (1 - factor), best_params['estimator__gamma'] * (1 + factor), steps)
        else:
            raise ValueError("Unsupported scoring method. Use 'accuracy' or 'tvl'.")

        best_score = search.best_score_
        print(f"Best score in iteration {i+1}: {best_score:.4f}")
        print(f"Best hyperparameters: {best_params}")

    return search.best_estimator_, best_params, best_score


def random_search_svm(X, y, n_iter=50, kernel='rbf', c_range=(-2, 2), gamma_range=(-4, 0),
                      degrees=None, scoring='accuracy', m=10, L=100, random_seed=42, n_jobs=-1):

    c_space     = loguniform(c_range[0], c_range[1])
    gamma_space = loguniform(gamma_range[0], gamma_range[1])
    d_space = np.arange(degrees[0], degrees[1] + 1) if degrees is not None else None

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    if scoring == 'accuracy':
        param_grid = ({'C': c_space, 'gamma': gamma_space, 'degree': d_space}
                      if kernel == 'poly' else
                      {'C': c_space, 'gamma': gamma_space})
        print("Accuracy tuning:")
        search = RandomizedSearchCV(SVC(kernel=kernel, probability=True, random_state=random_seed),
                                    param_grid, n_iter=n_iter, cv=cv_strategy, scoring='accuracy',
                                    n_jobs=n_jobs, verbose=3, random_state=random_seed)
        search.fit(X, y)
    elif scoring == 'tvl':
        param_grid = ({'estimator__C': c_space, 'estimator__gamma': gamma_space, 'estimator__degree': d_space}
                      if kernel == 'poly' else
                      {'estimator__C': c_space, 'estimator__gamma': gamma_space})
        print("TvL tuning:")
        talent_wrapper = TalentWrapper(estimator=SVC(kernel=kernel, probability=True, random_state=random_seed))
        objective_score = partial(tvl_scorer, m=m, L=L)
        search = RandomizedSearchCV(talent_wrapper, param_grid, n_iter=n_iter, cv=cv_strategy,
                                    scoring=objective_score, n_jobs=n_jobs, verbose=3, random_state=random_seed)
        search.fit(X, y)
    else:
        raise ValueError("Unsupported scoring method. Use 'accuracy' or 'tvl'.")

    best_estimator = search.best_estimator_
    best_params    = search.best_params_
    best_score     = search.best_score_
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")
    return best_estimator, best_params, best_score


def bayesian_optimization_svm(X, y, kernel='rbf', scoring='accuracy', n_trials=50,
                               m=10, L=100, random_seed=42):

    def objective(trial):
        c     = trial.suggest_float('C', 1e-2, 1e2, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e0, log=True)

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)
            model  = SVC(kernel=kernel, C=c, gamma=gamma, degree=degree,
                         probability=True, random_state=random_seed)
        else:
            model = SVC(kernel=kernel, C=c, gamma=gamma,
                        probability=True, random_state=random_seed)

        if scoring == 'accuracy':
            score = np.mean(cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy'))
        elif scoring == 'tvl':
            talent_wrapper = TalentWrapper(estimator=model)
            objective_score = partial(tvl_scorer, m=m, L=L)
            score = np.mean(cross_val_score(talent_wrapper, X, y, cv=cv_strategy, scoring=objective_score))
        else:
            raise ValueError("Unsupported scoring method. Use 'accuracy' or 'tvl'.")
        return score

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score  = study.best_value
    best_svc    = SVC(**best_params, kernel=kernel, probability=True, random_state=random_seed)
    best_svc.fit(X, y)
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")
    return best_svc, best_params, best_score
