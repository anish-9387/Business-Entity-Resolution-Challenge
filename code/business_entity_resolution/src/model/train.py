import numpy as np
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
from sklearn.ensemble import RandomForestClassifier

class ModelWrapper:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type  # 'lgb' or 'sklearn'
    
    def predict_proba(self, X):
        if self.model_type == 'lgb':
            return self.model.predict(X)  # LGB predict returns probabilities
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def feature_importance(self):
        if self.model_type == 'lgb':
            return self.model.feature_importance(importance_type='gain')
        else:
            return self.model.feature_importances_

def train_gbm(X_train, y_train, X_val, y_val):
    """
    Train a gradient boosted model. Returns a wrapper object with a .predict_proba(X) method
    that ALWAYS returns probabilities (not class labels).
    Uses LightGBM if available, else sklearn RandomForest.
    Handles class imbalance via scale_pos_weight or class_weight.
    """
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    print(f"Training stats: Positive class count: {pos_count}, Negative class count: {neg_count}")

    if HAS_LGB:
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': scale_pos_weight
        }
        
        print("Training LightGBM model...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        print(f"Best iteration: {model.best_iteration}")
        return ModelWrapper(model, 'lgb')
    else:
        print("LightGBM not available, training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            max_depth=15,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return ModelWrapper(model, 'sklearn')
