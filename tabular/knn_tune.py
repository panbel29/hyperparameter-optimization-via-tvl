import numpy as np
import optuna
from functools import partial
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from tvl_scorer import tvl_scorer, TalentWrapper


def grid_search_knn(X, y, kmin=1, kmax=51, scoring='accuracy', m=10, L=100, random_seed=42, n_jobs=-1):
    """
    Performs grid search for KNN.
    
    Args:
        X, y: Training data.
        scoring: Scoring method ('accuracy' or 'tvl').
    """
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    if scoring == 'accuracy':
        param_grid = {'n_neighbors': np.arange(kmin, kmax, 2),  # Odd values from kmin to kmax
                      'weights': ['uniform', 'distance']}
        print("Accuracy tuning:")
        search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv_strategy,
                              scoring='accuracy', n_jobs=n_jobs, verbose=3)
    elif scoring == 'tvl':
        param_grid = {'estimator__n_neighbors': np.arange(kmin, kmax, 2),
                      'estimator__weights': ['uniform', 'distance']}
        print("Tvl tuning:")
        talent_wrapper = TalentWrapper(KNeighborsClassifier())
        objective_scorer = partial(tvl_scorer, m=m, L=L)
        search = GridSearchCV(talent_wrapper, param_grid, cv=cv_strategy,
                              scoring=objective_scorer, n_jobs=n_jobs, verbose=3)
    else:
        raise ValueError("Unsupported scoring method. Use 'accuracy' or 'tvl'.")

    search.fit(X, y)
    best_estimator = search.best_estimator_
    best_score = search.best_score_
    best_params = search.best_params_
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")

    return best_estimator, best_params, best_score


def random_search_knn(X, y, n_iter=50, n_neighbors_range=(1, 51), weights=['uniform', 'distance'],
                      scoring='accuracy', m=10, L=100, random_seed=42, n_jobs=-1):

    n_neighbors_space = np.arange(n_neighbors_range[0], n_neighbors_range[1] + 1)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    if scoring == 'accuracy':
        param_dist = {'n_neighbors': n_neighbors_space, 'weights': weights}
        print("Random Search tuning:")
        search = RandomizedSearchCV(KNeighborsClassifier(), param_dist, n_iter=n_iter,
                                    cv=cv_strategy, scoring='accuracy', n_jobs=n_jobs,
                                    verbose=3, random_state=random_seed)
        search.fit(X, y)
    elif scoring == 'tvl':
        param_dist = {'estimator__n_neighbors': n_neighbors_space,
                      'estimator__weights': weights}
        print("Tvl tuning:")
        talent_wrapper = TalentWrapper(KNeighborsClassifier())
        objective_scorer = partial(tvl_scorer, m=m, L=L)
        search = RandomizedSearchCV(talent_wrapper, param_dist, n_iter=n_iter, cv=cv_strategy,
                                    scoring=objective_scorer, n_jobs=n_jobs,
                                    verbose=3, random_state=random_seed)
        search.fit(X, y)
    else:
        raise ValueError("Unsupported scoring method. Use 'accuracy' or 'tvl'.")

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")

    return best_estimator, best_params, best_score


def bayesian_knn(X, y, n_neighbors_range=(1, 51), weights=['uniform', 'distance'],
                 scoring='accuracy', n_trials=50, m=10, L=100, random_seed=42):

    def objective(trial):
        n_neighbors = trial.suggest_int('n_neighbors', n_neighbors_range[0], n_neighbors_range[1])
        selected_weight = trial.suggest_categorical('weights', weights)

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=selected_weight)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
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
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value
    best_rf = KNeighborsClassifier(**best_params)
    best_rf.fit(X, y)
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")

    return best_rf, best_params, best_score
