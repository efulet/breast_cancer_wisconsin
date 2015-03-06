"""
@created_at 2014-10-08
@author Exequiel Fuentes Lettura <efulet@gmail.com>
"""


import operator
import numpy as np


class RectangularClassifier:
    """
    RectangularClassifier class handles appropriate methods for fitting a 
    classifier using a rectangular model. This generates several rules which 
    shapes like a rectangle in the space.
    """
    
    def __init__(self):
        """
        RectangularClassifier constructor
        """
        self._X_train = None
        self._X_test = None
        self._rules = []
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier using a training set
        """
        X_train_dimension = X_train.shape[1]
        
        # Calculate the correlation
        self._X_train = np.array(X_train.todense())
        
        corrcoef_dict = {}
        
        for x in range(0, X_train_dimension):
            corrcoef_dict[x] = abs(np.corrcoef(y_train, self._X_train[:, x])[0][1])
        
        # Sort the correlation values in reverse order (max --> min)
        corrcoef_array_sorted = sorted(corrcoef_dict.items(), key=operator.itemgetter(1), reverse=True)
        
        # Start the training for generating the rules
        for rule_idx in range(0, len(corrcoef_array_sorted)):
            corrcoef_array_index = corrcoef_array_sorted[rule_idx][0]
            
            # Store the column with index and value
            X_train_col_dict = {}
            for i in range(0, len(self._X_train[:, corrcoef_array_index])):
                X_train_col_dict[i] = self._X_train[:, corrcoef_array_index][i]
            
            # Now, sort the values (min --> max)
            X_train_sorted_col = sorted(X_train_col_dict.items(), key=operator.itemgetter(1), reverse=False)
            
            # Initialize min_index to the first value in the X_train_sorted_col
            min_index = X_train_sorted_col[0][0]
            
            # Start with the min value until find 1 in y_train
            for i in range(0, len(X_train_sorted_col)/2):
                index = X_train_sorted_col[i][0]
                  
                if y_train[index] == 1:
                    min_index = index
                    break
            
            # Initialize max_index to the first value in the X_train_sorted_col
            max_index = len(X_train_sorted_col) - 1
            
            # Start with the min value until find 1 in y_train
            for i in range(len(X_train_sorted_col)-1, len(X_train_sorted_col)/2-1, -1):
                index = X_train_sorted_col[i][0]
                  
                if y_train[index] == 1:
                    max_index = index
                    break
            
            # Evaluate the rule
            sub_y_train_index = []
            add_to_sub_y_train = False
            
            for i in range(0, len(X_train_sorted_col)):
                index = X_train_sorted_col[i][0]
                if index == min_index: add_to_sub_y_train = True
                if add_to_sub_y_train:
                    sub_y_train_index.append(index)
                    if index == max_index: break
            
            if len(self._rules) > 0:
                if len(self._rules) == 1:
                    # Calculate current error
                    current_rejected = 0
                    y_train_indexs = self._rules[0][3]
                    for i in range(0, len(y_train_indexs)):
                        if y_train[y_train_indexs[i]] != 1: current_rejected += 1
                    
                    # Calculate the error with a new rule
                    new_rejected = 0
                    sub_y_train_index_intersect = set(self._rules[0][3]).intersection(sub_y_train_index)
                    sub_y_train_index_intersect = list(sub_y_train_index_intersect)
                    
                    for i in range(0, len(sub_y_train_index_intersect)):
                        if y_train[sub_y_train_index_intersect[i]] != 1: new_rejected += 1
                    
                    # If better, then add to the rules
                    if new_rejected < current_rejected:
                        self._rules.append([corrcoef_array_index, min_index, max_index, sub_y_train_index])
                else:
                    # Intersect all previous rules
                    sub_y_train_index_intersect = set(self._rules[0][3]).intersection(self._rules[0][3])
                    
                    for i in range(1, len(self._rules)):
                        rule = self._rules[i]
                        sub_y_train_index_intersect = set(sub_y_train_index_intersect).intersection(rule[3])
                    
                    sub_y_train_index_intersect = list(sub_y_train_index_intersect)
                    
                    # Calculate current error
                    current_rejected = 0
                    for i in range(0, len(sub_y_train_index_intersect)):
                        if y_train[sub_y_train_index_intersect[i]] != 1: current_rejected += 1
                    
                    # Calculate the error with a new rule
                    new_rejected = 0
                    sub_y_train_index_intersect = set(sub_y_train_index_intersect).intersection(sub_y_train_index)
                    sub_y_train_index_intersect = list(sub_y_train_index_intersect)
                    
                    for i in range(0, len(sub_y_train_index_intersect)):
                        if y_train[sub_y_train_index_intersect[i]] != 1: new_rejected += 1
                    
                    # If better, then add to the rules
                    if new_rejected < current_rejected:
                        self._rules.append([corrcoef_array_index, min_index, max_index, sub_y_train_index])
            else:
                self._rules.append([corrcoef_array_index, min_index, max_index, sub_y_train_index])
        
        # Return this class object
        return self
    
    def _evaluate_rules(self, X_train, y_train):
        """
        Evaluate the rules against training set
        """
        identified = 0
        rejected = 0
        
        for i in range(0, len(X_train)):
            rules_list = []
            for j in range(0, len(self._rules)):
                rule = self._rules[j]
                index = rule[0]
                min_index = rule[1]
                max_index = rule[2]
                str = "(X_train[" + `i` + "][" + `index` + "] >= X_train[" + \
                    `min_index` + "][" + `index` + "] and X_train[" + `i` + "][" + \
                    `index` + "] <= X_train[" + `max_index` + "][" + `index` + "])"
                rules_list.append(str)
            
            if eval(' and '.join(rules_list)):
                if y_train[i] == 1: identified += 1
                else: rejected += 1
        
        return identified, rejected
    
    def predict(self, X_test):
        """
        Generate a prediction set using rules
        """
        X_test = np.array(X_test.todense())
        y_pred = []
        
        for i in range(0, len(X_test)):
            rules_list = []
            for j in range(0, len(self._rules)):
                rule = self._rules[j]
                index = rule[0]
                min_index = rule[1]
                max_index = rule[2]
                str = "(X_test[" + `i` + "][" + `index` + "] >= self._X_train[" + \
                    `min_index` + "][" + `index` + "] and X_test[" + `i` + "][" + \
                    `index` + "] <= self._X_train[" + `max_index` + "][" + `index` + "])"
                rules_list.append(str)
             
            if eval(' and '.join(rules_list)):
                y_pred.append(1)
            else:
                y_pred.append(0)
        
        return np.array(y_pred)
