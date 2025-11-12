# Machine Learning Assignment - Decision Trees, Random Forest & AdaBoost
# Implemented from scratch for educational purposes

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# =============================================================================
# DECISION TREE CLASSIFIER - From Scratch Implementation
# =============================================================================

class DecisionTreeNode:
    """Node class for decision tree - represents either a decision point or leaf"""
    def __init__(self):
        self.feature_idx = None      # Feature index for split
        self.threshold = None        # Split threshold
        self.left = None            # Left child
        self.right = None           # Right child
        self.value = None           # Predicted class (leaf nodes)
        self.is_leaf = False        # Leaf indicator

class DecisionTreeClassifier:
    """Decision Tree Classifier using Gini impurity for splits"""
    
    def __init__(self, max_depth=None, min_samples_leaf=1, random_state=42, max_features=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_features = max_features
        self.root = None
        self.classes_ = None
        self.n_features_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity: 1 - Σ(p_i)²"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _information_gain(self, y, left_y, right_y):
        """Calculate information gain from split"""
        n = len(y)
        n_left, n_right = len(left_y), len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_gini = self._gini_impurity(y)
        weighted_child_gini = (n_left / n) * self._gini_impurity(left_y) + \
                             (n_right / n) * self._gini_impurity(right_y)
        
        return parent_gini - weighted_child_gini
    
    def _best_split(self, X, y):
        """Find best feature and threshold for splitting"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Feature selection for Random Forest
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                gain = self._information_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build decision tree"""
        node = DecisionTreeNode()
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(np.unique(y)) == 1 or \
           len(y) < 2 * self.min_samples_leaf:
            
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature is None or best_gain == 0:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Train the decision tree"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        
        # Create label mapping
        self.label_to_int = {label: i for i, label in enumerate(self.classes_)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        
        y_int = np.array([self.label_to_int[label] for label in y])
        
        self.root = self._build_tree(X, y_int)
        return self
    
    def _predict_sample(self, x):
        """Predict single sample"""
        node = self.root
        
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return self.int_to_label[node.value]
    
    def predict(self, X):
        """Predict multiple samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return np.array([self._predict_sample(x) for x in X])

# =============================================================================
# RANDOM FOREST CLASSIFIER - From Scratch Implementation
# =============================================================================

class RandomForestClassifier:
    """Random Forest using bootstrap sampling and feature randomness"""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, 
                 max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        self.n_features_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[bootstrap_indices], y[bootstrap_indices]
    
    def _get_max_features(self, n_features):
        """Calculate number of features per split"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features is None:
            return n_features
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def fit(self, X, y):
        """Train Random Forest"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        
        max_features_per_tree = self._get_max_features(self.n_features_)
        self.trees = []
        
        print(f"Training Random Forest with {self.n_estimators} trees...")
        
        for i in range(self.n_estimators):
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features_per_tree,
                random_state=self.random_state + i if self.random_state is not None else None
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Trained {i + 1}/{self.n_estimators} trees")
        
        print("Random Forest training completed!")
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_samples = X.shape[0]
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        predictions = []
        for i in range(n_samples):
            sample_votes = tree_predictions[:, i]
            vote_counts = Counter(sample_votes)
            majority_class = vote_counts.most_common(1)[0][0]
            predictions.append(majority_class)
        
        return np.array(predictions)

# =============================================================================
# ADABOOST CLASSIFIER - From Scratch Implementation  
# =============================================================================

class AdaBoostClassifier:
    """AdaBoost using decision stumps as weak learners"""
    
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
        self.n_features_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _make_estimator(self):
        """Create decision stump (depth=1 tree)"""
        return DecisionTreeClassifier(
            max_depth=1,
            min_samples_leaf=1,
            random_state=self.random_state
        )
    
    def fit(self, X, y):
        """Train AdaBoost using SAMME algorithm"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Label encoding
        self.label_to_int = {label: i for i, label in enumerate(self.classes_)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        
        y_int = np.array([self.label_to_int[label] for label in y])
        
        # Initialize uniform sample weights
        sample_weight = np.ones(n_samples) / n_samples
        
        print(f"Training AdaBoost with {self.n_estimators} weak learners...")
        
        self.estimators_ = []
        self.estimator_weights_ = []
        
        for iboost in range(self.n_estimators):
            sample_weight, estimator_weight, estimator_error = self._boost(
                X, y_int, sample_weight, iboost
            )
            
            if sample_weight is None:
                break
                
            self.estimator_weights_.append(estimator_weight)
            
            if estimator_error == 0:
                break
                
            if (iboost + 1) % 10 == 0 or iboost == 0:
                print(f"  Trained {iboost + 1}/{self.n_estimators} estimators, "
                      f"error: {estimator_error:.3f}, weight: {estimator_weight:.3f}")
        
        print("AdaBoost training completed!")
        return self
    
    def _boost(self, X, y, sample_weight, iboost):
        """Single boosting step"""
        estimator = self._make_estimator()
        
        n_samples = X.shape[0]
        
        # Weighted bootstrap sample
        weighted_indices = np.random.choice(
            n_samples, 
            size=n_samples, 
            replace=True, 
            p=sample_weight / np.sum(sample_weight)
        )
        
        X_weighted = X[weighted_indices]
        y_weighted = y[weighted_indices]
        
        y_weighted_orig = np.array([self.int_to_label[yi] for yi in y_weighted])
        estimator.fit(X_weighted, y_weighted_orig)
        
        # Predictions on original training set
        y_pred_orig = estimator.predict(X)
        y_pred = np.array([self.label_to_int[pred] for pred in y_pred_orig])
        
        # Calculate weighted error
        incorrect = y_pred != y
        estimator_error = np.average(incorrect, weights=sample_weight)
        
        # Check if error is too high
        if estimator_error >= 1.0 - (1.0 / self.n_classes_):
            return None, None, None
        
        if estimator_error <= 0:
            estimator_error = 1e-10
        
        # Calculate estimator weight (SAMME algorithm)
        if self.n_classes_ == 2:
            estimator_weight = self.learning_rate * 0.5 * np.log(
                (1.0 - estimator_error) / estimator_error
            )
        else:
            estimator_weight = self.learning_rate * np.log(
                (1.0 - estimator_error) / estimator_error
            ) + np.log(self.n_classes_ - 1.0)
        
        self.estimators_.append(estimator)
        
        # Update sample weights
        if iboost < self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight * incorrect)
            sample_weight /= np.sum(sample_weight)
            
            if np.sum(sample_weight) == 0:
                sample_weight = np.ones(len(sample_weight)) / len(sample_weight)
        
        return sample_weight, estimator_weight, estimator_error
    
    def predict(self, X):
        """Predict using weighted voting"""
        decision = self.decision_function(X)
        return self.classes_.take(np.argmax(decision, axis=1))
    
    def decision_function(self, X):
        """Compute weighted votes for each class"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_samples = X.shape[0]
        decision = np.zeros((n_samples, self.n_classes_))
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            current_pred = estimator.predict(X)
            
            for i, pred in enumerate(current_pred):
                class_idx = self.label_to_int[pred]
                decision[i, class_idx] += weight
        
        return decision

# =============================================================================
# EVALUATION METRICS - From Scratch
# =============================================================================

def accuracy_score(y_true, y_pred):
    """Calculate accuracy score"""
    return np.mean(y_true == y_pred)

def f1_score(y_true, y_pred, average='macro'):
    """Calculate F1 score"""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if average == 'macro':
        f1_scores = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    return np.mean(y_true == y_pred)

def classification_report(y_true, y_pred):
    """Generate detailed classification report"""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    report = "              precision    recall  f1-score   support\n\n"
    
    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        total_support += support
        
        report += f"{str(cls):>12} {precision:>9.2f} {recall:>9.2f} {f1:>9.2f} {support:>9}\n"
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    if total_support > 0:
        weighted_precision /= total_support
        weighted_recall /= total_support
        weighted_f1 /= total_support
    
    report += "\n"
    report += f"    accuracy                     {accuracy:>9.2f} {total_support:>9}\n"
    report += f"   macro avg {weighted_precision:>9.2f} {weighted_recall:>9.2f} {macro_f1:>9.2f} {total_support:>9}\n"
    report += f"weighted avg {weighted_precision:>9.2f} {weighted_recall:>9.2f} {weighted_f1:>9.2f} {total_support:>9}\n"
    
    return report

# =============================================================================
# MAIN EXPERIMENT - Following Assignment Requirements
# =============================================================================

def main():
    """Main function to run the complete experiment"""
    
    print("="*60)
    print("MACHINE LEARNING ASSIGNMENT")
    print("Decision Trees, Random Forest & AdaBoost")
    print("="*60)
    
    # Load your dataset (replace 'Data.csv' with your actual file)
    try:
        data = pd.read_csv('Data.csv')
        print(f"Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
    except FileNotFoundError:
        print("Error: Data.csv not found. Please ensure your dataset is in the same directory.")
        print("Creating sample data for demonstration...")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples, n_features = 1000, 10
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(['Class_A', 'Class_B', 'Class_C'], n_samples)
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        data['target'] = y
    
    # Prepare features and target
    target_column = 'personality_type' if 'personality_type' in data.columns else data.columns[-1]
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    print(f"Target variable: {target_column}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution:")
    for cls in np.unique(y):
        count = np.sum(y == cls)
        print(f"  {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split data: 60% train, 30% validation, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"Training: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
    
    # 1. Train Decision Tree
    print("\n" + "="*60)
    print("STEP 1: Training Decision Tree")
    print("="*60)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    
    val_pred_dt = dt.predict(X_val)
    dt_val_acc = accuracy_score(y_val, val_pred_dt)
    dt_val_f1 = f1_score(y_val, val_pred_dt, average='macro')
    
    print(f"Decision Tree - Validation Accuracy: {dt_val_acc:.3f}")
    print(f"Decision Tree - Validation F1: {dt_val_f1:.3f}")
    
    # 2. Train Random Forest
    print("\n" + "="*60)
    print("STEP 2: Training Random Forest")
    print("="*60)
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    val_pred_rf = rf.predict(X_val)
    rf_val_acc = accuracy_score(y_val, val_pred_rf)
    rf_val_f1 = f1_score(y_val, val_pred_rf, average='macro')
    
    print(f"Random Forest - Validation Accuracy: {rf_val_acc:.3f}")
    print(f"Random Forest - Validation F1: {rf_val_f1:.3f}")
    
    # 3. Train AdaBoost
    print("\n" + "="*60)
    print("STEP 3: Training AdaBoost")
    print("="*60)
    ada = AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    ada.fit(X_train, y_train)
    
    val_pred_ada = ada.predict(X_val)
    ada_val_acc = accuracy_score(y_val, val_pred_ada)
    ada_val_f1 = f1_score(y_val, val_pred_ada, average='macro')
    
    print(f"AdaBoost - Validation Accuracy: {ada_val_acc:.3f}")
    print(f"AdaBoost - Validation F1: {ada_val_f1:.3f}")
    
    # 4. Compare models on validation set
    print("\n" + "="*60)
    print("STEP 4: Model Comparison on Validation Set")
    print("="*60)
    
    print("VALIDATION RESULTS:")
    print("-" * 40)
    print(f"Decision Tree - Accuracy: {dt_val_acc:.3f}, F1: {dt_val_f1:.3f}")
    print(f"Random Forest - Accuracy: {rf_val_acc:.3f}, F1: {rf_val_f1:.3f}")
    print(f"AdaBoost      - Accuracy: {ada_val_acc:.3f}, F1: {ada_val_f1:.3f}")
    
    # Determine best model
    val_scores = {
        'Decision Tree': dt_val_acc,
        'Random Forest': rf_val_acc,
        'AdaBoost': ada_val_acc
    }
    
    best_model_name = max(val_scores.keys(), key=lambda k: val_scores[k])
    print(f"\nBest model on validation: {best_model_name}")
    
    # 5. Final evaluation on test set
    print("\n" + "="*60)
    print("STEP 5: Final Evaluation on Test Set")
    print("="*60)
    
    # Retrain on train+validation data
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    
    # Final models
    final_dt = DecisionTreeClassifier(random_state=42)
    final_dt.fit(X_trainval, y_trainval)
    
    final_rf = RandomForestClassifier(
        n_estimators=50, max_depth=12, max_features='sqrt',
        min_samples_leaf=2, random_state=42
    )
    final_rf.fit(X_trainval, y_trainval)
    
    final_ada = AdaBoostClassifier(
        n_estimators=50, learning_rate=1.0, random_state=42
    )
    final_ada.fit(X_trainval, y_trainval)
    
    # Test set predictions
    test_pred_dt = final_dt.predict(X_test)
    test_pred_rf = final_rf.predict(X_test)
    test_pred_ada = final_ada.predict(X_test)
    
    # Test set metrics
    dt_test_acc = accuracy_score(y_test, test_pred_dt)
    rf_test_acc = accuracy_score(y_test, test_pred_rf)
    ada_test_acc = accuracy_score(y_test, test_pred_ada)
    
    dt_test_f1 = f1_score(y_test, test_pred_dt, average='macro')
    rf_test_f1 = f1_score(y_test, test_pred_rf, average='macro')
    ada_test_f1 = f1_score(y_test, test_pred_ada, average='macro')
    
    print("FINAL TEST RESULTS:")
    print("-" * 40)
    print(f"Decision Tree - Test Accuracy: {dt_test_acc:.3f}, F1: {dt_test_f1:.3f}")
    print(f"Random Forest - Test Accuracy: {rf_test_acc:.3f}, F1: {rf_test_f1:.3f}")
    print(f"AdaBoost      - Test Accuracy: {ada_test_acc:.3f}, F1: {ada_test_f1:.3f}")
    
    # Determine best model on test set
    test_scores = {
        'Decision Tree': dt_test_acc,
        'Random Forest': rf_test_acc,
        'AdaBoost': ada_test_acc
    }
    
    best_test_model = max(test_scores.keys(), key=lambda k: test_scores[k])
    best_test_acc = test_scores[best_test_model]
    
    print(f"\nBest model on test set: {best_test_model}")
    
    if best_test_model != 'Decision Tree':
        improvement = ((best_test_acc - dt_test_acc) / dt_test_acc) * 100
        print(f"Improvement over Decision Tree: {improvement:.1f}%")
    
    # Detailed classification report for best model
    print(f"\nDetailed Classification Report - {best_test_model}:")
    print("-" * 60)
    if best_test_model == 'Decision Tree':
        print(classification_report(y_test, test_pred_dt))
    elif best_test_model == 'Random Forest':
        print(classification_report(y_test, test_pred_rf))
    else:
        print(classification_report(y_test, test_pred_ada))
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Algorithm insights
    print("\nALGORITHM INSIGHTS:")
    print("-" * 25)
    print("Decision Tree:")
    print("  • Single tree with greedy splits")
    print("  • Fast training and prediction") 
    print("  • Prone to overfitting")
    print("  • High interpretability")
    
    print("\nRandom Forest:")
    print("  • Ensemble of decision trees")
    print("  • Bootstrap sampling + feature randomness")
    print("  • Reduces overfitting through averaging")
    print(f"  • Uses {rf.n_estimators} trees")
    
    print("\nAdaBoost:")
    print("  • Sequential boosting ensemble")
    print("  • Adaptive sample weighting")
    print("  • Focus on hard examples")
    print(f"  • Uses {len(ada.estimators_)} decision stumps")

if __name__ == "__main__":
    main()