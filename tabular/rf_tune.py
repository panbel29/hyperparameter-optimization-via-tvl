import numpy as np
import optuna
from functools import partial
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from tvl_scorer import tvl_scorer, TalentWrapper


def grid_search_rf(X, y, iterations=3,
                   n_est_init=(10, 100), max_depth_init=(5, 50), min_samples_split_init=(2, 11),
                   scoring='accuracy', m=10, L=100, random_seed=42, n_jobs=-1):
    """
    Performs coarse-to-fine grid search for Random Forest.
    """
    bounds = {
        'n_estimators':      list(n_est_init),
        'max_depth':         list(max_depth_init),
        'min_samples_split': list(min_samples_split_init)
    }

    best_params = None
    best_score  = -np.inf
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    prefix = 'estimator__' if scoring == 'tvl' else ''

    for i in range(iterations):
        print(f"\n--- Iteration {i+1} ---")

        curr_n_est_space            = np.linspace(bounds['n_estimators'][0],      bounds['n_estimators'][1],      4, dtype=int)
        curr_max_depth_space        = np.linspace(bounds['max_depth'][0],          bounds['max_depth'][1],          4, dtype=int)
        curr_min_samples_split_space = np.unique(np.linspace(bounds['min_samples_split'][0], bounds['min_samples_split'][1], 4, dtype=int))

        param_grid = {
            f'{prefix}n_estimators':      curr_n_est_space,
            f'{prefix}max_depth':         curr_max_depth_space,
            f'{prefix}min_samples_split': curr_min_samples_split_space
        }
        print(f"Grid: {param_grid}")

        if scoring == 'accuracy':
            model          = RandomForestClassifier(random_state=random_seed)
            current_scorer = 'accuracy'
        elif scoring == 'tvl':
            model          = TalentWrapper(RandomForestClassifier(random_state=random_seed))
            current_scorer = partial(tvl_scorer, m=m, L=L)

        search = GridSearchCV(model, param_grid, cv=cv_strategy,
                              scoring=current_scorer, n_jobs=n_jobs, verbose=3)
        search.fit(X, y)

        best_params = search.best_params_
        best_score  = search.best_score_
        print(f"Best Score: {best_score:.4f}")
        print(f"Best Params: {best_params}")

        if i < iterations - 1:
            for key in ['n_estimators', 'max_depth', 'min_samples_split']:
                val          = best_params[f"{prefix}{key}"]
                current_range = bounds[key][1] - bounds[key][0]
                step         = current_range / 3

                new_min = max(n_est_init[0] if key == 'n_estimators' else
                              max_depth_init[0] if key == 'max_depth' else
                              min_samples_split_init[0], int(val - step / 2))
                new_max = min(n_est_init[1] if key == 'n_estimators' else
                              max_depth_init[1] if key == 'max_depth' else
                              min_samples_split_init[1], int(val + step / 2))

                if new_min == new_max:
                    new_max += 2
                bounds[key] = [new_min, new_max]

    return search.best_estimator_, best_params, best_score


def random_search_rf(X, y, n_iter=50, n_est_range=(10, 100), max_depth_range=(5, 50),
                     min_samples_split_range=(2, 11), scoring='accuracy', m=10, L=100,
                     random_seed=42, n_jobs=-1):

    n_est_space            = np.arange(n_est_range[0],            n_est_range[1] + 1)
    max_depth_space        = np.arange(max_depth_range[0],        max_depth_range[1] + 1)
    min_samples_split_space = np.arange(min_samples_split_range[0], min_samples_split_range[1] + 1)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    if scoring == 'accuracy':
        param_dist = {
            'n_estimators':      n_est_space,
            'max_depth':         max_depth_space,
            'min_samples_split': min_samples_split_space
        }
        print("Random Search tuning:")
        search = RandomizedSearchCV(RandomForestClassifier(random_state=random_seed),
                                    param_dist, n_iter=n_iter, cv=cv_strategy,
                                    scoring='accuracy', n_jobs=n_jobs,
                                    verbose=3, random_state=random_seed)
        search.fit(X, y)
    elif scoring == 'tvl':
        param_dist = {
            'estimator__n_estimators':      n_est_space,
            'estimator__max_depth':         max_depth_space,
            'estimator__min_samples_split': min_samples_split_space
        }
        print("Tvl tuning:")
        talent_wrapper  = TalentWrapper(RandomForestClassifier(random_state=random_seed))
        objective_scorer = partial(tvl_scorer, m=m, L=L)
        search = RandomizedSearchCV(talent_wrapper, param_dist, n_iter=n_iter, cv=cv_strategy,
                                    scoring=objective_scorer, n_jobs=n_jobs,
                                    verbose=3, random_state=random_seed)
        search.fit(X, y)
    else:
        raise ValueError("Unsupported scoring method. Use 'accuracy' or 'tvl'.")

    best_estimator = search.best_estimator_
    best_params    = search.best_params_
    best_score     = search.best_score_
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")
    return best_estimator, best_params, best_score


def halving_grid_search_rf(X, y, scoring='accuracy', m=10, L=100, random_seed=42, n_jobs=-1):

    n_est_space            = np.arange(10, 101, 10)
    max_depth_space        = np.arange(5, 51, 5)
    min_samples_split_space = np.arange(2, 12, 2)
    prefix = 'estimator__' if scoring == 'tvl' else ''

    param_grid = {
        f'{prefix}n_estimators':      n_est_space,
        f'{prefix}max_depth':         max_depth_space,
        f'{prefix}min_samples_split': min_samples_split_space
    }

    if scoring == 'accuracy':
        model          = RandomForestClassifier(random_state=random_seed)
        current_scorer = 'accuracy'
    elif scoring == 'tvl':
        model          = TalentWrapper(RandomForestClassifier(random_state=random_seed))
        current_scorer = partial(tvl_scorer, m=m, L=L)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    search = HalvingGridSearchCV(model, param_grid, factor=3, cv=cv_strategy,
                                 scoring=current_scorer, n_jobs=n_jobs, verbose=3)
    search.fit(X, y)

    best_estimator = search.best_estimator_
    best_params    = search.best_params_
    best_score     = search.best_score_
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")
    return best_estimator, best_params, best_score


def bayesian_rf(X, y, scoring='accuracy', n_trials=50, m=10, L=100, random_seed=42):

    def objective(trial):
        n_estimators      = trial.suggest_int('n_estimators', 10, 100)
        max_depth         = trial.suggest_int('max_depth', 5, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 11)

        model       = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                             min_samples_split=min_samples_split, random_state=random_seed)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

        if scoring == 'accuracy':
            score = np.mean(cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy'))
        elif scoring == 'tvl':
            talent_wrapper  = TalentWrapper(estimator=model)
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
    best_rf     = RandomForestClassifier(**best_params, random_state=random_seed)
    best_rf.fit(X, y)
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")
    return best_rf, best_params, best_score
