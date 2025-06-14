import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

def model_pipeline(
    df: pd.DataFrame,
    model_factory,            # e.g. lambda **p: RandomForestClassifier(**p)
    non_numerical_columns=None,
    normalize='global',        # 'global', 'per_participant', or None
    test_val_split=0.1,
    param_distributions=None,  # dict of {param: tuple or list or optuna Distribution}
    n_trials=50,               # how many Optuna trials
    search_cv=3,               # inner CV folds
    test_participant=None,     # if set, only this participant is used as test/validation
    use_rfe: bool = True       # whether to do backward‐selection before fitting
):
    df = df.copy()

    # 1) Identify columns
    if non_numerical_columns is None:
        non_numerical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    numerical_columns = [c for c in df.columns if c not in non_numerical_columns]

    # 2) Normalize
    if normalize == 'global':
        for c in numerical_columns:
            df[c] = (df[c] - df[c].mean()) / df[c].std()
    elif normalize == 'per_participant':
        df[numerical_columns] = (
            df.groupby('participant')[numerical_columns]
              .transform(lambda x: (x - x.mean())/x.std())
        )

    # 3) Encode categoricals
    for c in df.columns:
        if df[c].dtype == 'object' and c not in ['participant', 'timestamp', 'genre']:
            df[c] = LabelEncoder().fit_transform(df[c])

    # 4) Encode target
    genre_le = LabelEncoder()
    df['genre_encoded'] = genre_le.fit_transform(df['genre'])

    # 5) Build feature list & participant list
    features = [c for c in df.columns
                if c not in ['participant','timestamp','genre','genre_encoded']]
    participants = ([test_participant] if test_participant
                    else df['participant'].unique())

    # 6) Turn your raw specs into Optuna distributions
    optuna_dist = {}
    for name,spec in (param_distributions or {}).items():
        if isinstance(spec, tuple) and len(spec)==3 and all(isinstance(x,int) for x in spec):
            low,high,step = spec
            optuna_dist[name] = IntDistribution(low=low, high=high, step=step)
        elif isinstance(spec, tuple) and len(spec)==3 and any(isinstance(x,float) for x in spec):
            low,high,q = spec
            optuna_dist[name] = FloatDistribution(low=low, high=high, step=q)
        elif isinstance(spec, list):
            optuna_dist[name] = CategoricalDistribution(spec)
        elif hasattr(spec,'__class__') and 'Distribution' in spec.__class__.__name__:
            optuna_dist[name] = spec
        else:
            raise ValueError(f"Cannot interpret {name} spec={spec}")

    results = {
      'reports':{}, 'macro_f1s':[], 'feature_importances':{}, 'avg_macro_f1':None
    }

    # 7) Loop over each LOPO fold
    for test_p in participants:
        print(f"\n--- Testing on {test_p} ---")
        train_df = df[df['participant'] != test_p]
        test_df  = df[df['participant'] == test_p]

        X_train, y_train = train_df[features], train_df['genre_encoded']
        X_test_full, y_test_full = test_df[features], test_df['genre_encoded']

        # 8) carve off a 10% validation from your hold‐out
        X_test, X_val, y_test, y_val = train_test_split(
            X_test_full, y_test_full,
            test_size=test_val_split,
            stratify=y_test_full,
            random_state=42
        )

        # 9) Build your Pipeline steps
        steps = []
        if use_rfe:
            # RFECV will do backward selection **inside** each cv‐fold
            steps.append((
              'feature_selection',
              RFECV(
                estimator = model_factory(),
                step      = 1,
                cv        = search_cv,
                scoring   = 'f1_macro',
                n_jobs    = -1,
                min_features_to_select=10
              )
            ))
        # final estimator named 'clf' so we can prefix hyperparams with 'clf__'
        steps.append(('clf', model_factory()))

        pipe = Pipeline(steps)

        # 10) Optuna inside the pipeline
        optuna_cv = OptunaSearchCV(
            pipe,
            param_distributions=optuna_dist,
            cv=search_cv,
            scoring='f1_macro',
            n_trials=n_trials,
            random_state=42,
            n_jobs=-1
        )
        optuna_cv.fit(X_train, y_train)

        print("Optuna best params:", optuna_cv.best_params_)
        best_model = optuna_cv.best_estimator_

        # 11) Test set performance
        y_pred = best_model.predict(X_test)
        print(classification_report(y_test, y_pred,
              target_names=genre_le.classes_, zero_division=0))

        # 12) Feature importances (if available)
        if hasattr(best_model, "named_steps") and 'clf' in best_model.named_steps:
            m = best_model.named_steps['clf']
            if hasattr(m, "feature_importances_"):
                imp = m.feature_importances_
                # Get mask of selected features from RFECV
                selected_mask = best_model.named_steps['feature_selection'].get_support()
                selected_features = np.array(features)[selected_mask]
                # Align feature names with importances
                df_imp = (pd.DataFrame({'feature': selected_features, 'importance': imp})
                            .sort_values('importance', ascending=False))
                results['feature_importances'][test_p] = df_imp

        # 13) record metrics
        f1 = f1_score(y_test, y_pred, average='macro')
        results['macro_f1s'].append(f1)
        results['reports'][test_p] = {
          'f1_macro': f1,
          'test_report': classification_report(
               y_test, y_pred, target_names=genre_le.classes_,
               output_dict=True, zero_division=0
          )
        }

        # 14) Validation set performance
        y_val_pred = best_model.predict(X_val)
        print("Validation performance:")
        print(classification_report(y_val, y_val_pred,
              target_names=genre_le.classes_, zero_division=0))
        results['reports'][test_p]['val_report'] = classification_report(
            y_val, y_val_pred,
            target_names=genre_le.classes_,
            output_dict=True, zero_division=0
        )

    results['avg_macro_f1'] = np.mean(results['macro_f1s'])
    print(f"\n=== Avg Macro F1: {results['avg_macro_f1']:.3f} ===")
    return results
