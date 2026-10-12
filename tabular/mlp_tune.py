import numpy as np
import optuna
from functools import partial
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from tvl_scorer import tvl_scorer, TalentWrapper


def grid_search_mlp(X, y, iterations=3, scoring='accuracy', m=10, L=100, random_seed=42, n_jobs=-1):

    # Initial search bounds
    lr_bounds       = [-4.0, -1.0]   # Log10 scale: 10^-4 to 10^-1
    max_iter_bounds = [50, 300]
    batch_options   = [8, 16, 32, 64, 128]

    best_params = None
    prefix = 'estimator__' if scoring == 'tvl' else ''

    for i in range(iterations):
        print(f"\n--- MLP Iteration {i+1} ---")

        curr_lr_space       = np.logspace(lr_bounds[0], lr_bounds[1], 4)
        curr_max_iter_space = np.linspace(max_iter_bounds[0], max_iter_bounds[1], 4, dtype=int)
        cv_strategy         = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

        # Full batch list on first iteration; narrow to neighbours of best in subsequent ones
        if i == 0:
            curr_batch_space = [8, 16, 32, 64]
        else:
            idx = batch_options.index(best_params[f'{prefix}batch_size'])
            curr_batch_space = batch_options[max(0, idx - 1):min(len(batch_options), idx + 2)]

        param_grid = {
            f'{prefix}learning_rate_init': curr_lr_space,
            f'{prefix}max_iter':           curr_max_iter_space,
            f'{prefix}batch_size':         curr_batch_space
        }
        print(f"Searching in: {param_grid}")

        if scoring == 'accuracy':
            model          = MLPClassifier(random_state=random_seed)
            current_scorer = 'accuracy'
        elif scoring == 'tvl':
            model          = TalentWrapper(MLPClassifier(random_state=random_seed))
            current_scorer = partial(tvl_scorer, m=m, L=L)

        search = GridSearchCV(model, param_grid, cv=cv_strategy,
                              scoring=current_scorer, n_jobs=n_jobs, verbose=3)
        search.fit(X, y)

        best_params = search.best_params_
        print(f"Iteration {i+1} Best: {best_params} | Score: {search.best_score_:.4f}")

        # Narrow bounds around the best value for the next iteration
        if i < iterations - 1:
            current_lr_log = np.log10(best_params[f'{prefix}learning_rate_init'])
            lr_step        = (lr_bounds[1] - lr_bounds[0]) / 4
            lr_bounds      = [max(-4.0, current_lr_log - lr_step), min(-1.0, current_lr_log + lr_step)]

            current_mi      = best_params[f'{prefix}max_iter']
            mi_step         = (max_iter_bounds[1] - max_iter_bounds[0]) / 4
            max_iter_bounds = [max(50, int(current_mi - mi_step)), min(300, int(current_mi + mi_step))]

    return search.best_estimator_, best_params, search.best_score_


def random_search_mlp(X, y, n_iter=50, scoring='accuracy', m=10, L=100, random_seed=42, n_jobs=-1):

    learning_rate_space = loguniform(1e-4, 1e-1)
    batch_size_space    = [8, 16, 32, 64, 128]
    max_iter_space      = np.arange(50, 301)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    if scoring == 'accuracy':
        param_dist = {
            'learning_rate_init': learning_rate_space,
            'batch_size':         batch_size_space,
            'max_iter':           max_iter_space
        }
        print("Random Search tuning:")
        search = RandomizedSearchCV(MLPClassifier(random_state=random_seed),
                                    param_dist, n_iter=n_iter, cv=cv_strategy,
                                    scoring='accuracy', n_jobs=n_jobs,
                                    verbose=3, random_state=random_seed)
        search.fit(X, y)
    elif scoring == 'tvl':
        param_dist = {
            'estimator__learning_rate_init': learning_rate_space,
            'estimator__batch_size':         batch_size_space,
            'estimator__max_iter':           max_iter_space
        }
        print("Tvl tuning:")
        talent_wrapper   = TalentWrapper(MLPClassifier(random_state=random_seed))
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


def bayesian_mlp(X, y, scoring='accuracy', n_trials=50, m=10, L=100, random_seed=42):

    def objective(trial):
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)
        batch_size         = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        max_iter           = trial.suggest_int('max_iter', 50, 300)

        model       = MLPClassifier(learning_rate_init=learning_rate_init,
                                    batch_size=batch_size, max_iter=max_iter,
                                    random_state=random_seed)
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
    best_mlp    = MLPClassifier(**best_params, random_state=random_seed)
    best_mlp.fit(X, y)
    print(f"Best score: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")
    return best_mlp, best_params, best_score
