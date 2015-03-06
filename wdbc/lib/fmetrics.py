"""
@created_at 2014-10-08
@author Exequiel Fuentes Lettura <efulet@gmail.com>
"""


from sklearn import metrics


class FMetrics:
    """
    Encapsulate metrics for reporting via console
    """
    
    def __init__(self):
        """
        """
    
    def apparent_error_rate(self, y_train, y_train_pred):
        """
        Calculate Apparent error rate
        """
        error = 0
        for i in range(0, len(y_train_pred)):
            if y_train[i] != y_train_pred[i]: error += 1
        
        return (float(error)/len(y_train_pred))
    
    def report(self, y_train, y_train_pred, y_test, y_pred):
        """
        Create a report with several metrics
        """
        # **********************************************************************
        # Tasa de error aparente: tasa de error obtenida al clasificar las mismas 
        # instancias de entrenamiento.
        print "Apparent error rate: %4f" % self.apparent_error_rate(y_train, y_train_pred)
        
        # **********************************************************************
        # The mean_absolute_error function computes the mean absolute error, 
        # which is a risk function corresponding to the expected value of the 
        # absolute error loss.
        # Tasa de error verdadera: probabilidad de clasificar incorrectamente nuevos 
        # casos. Para ello se utiliza el conjunto de datos de prueba.
        print "Mean absolute error: %.4f" % metrics.mean_absolute_error(y_test, y_pred)
        
        # **********************************************************************
        # The confusion_matrix function computes the confusion matrix to 
        # evaluate the accuracy on a classification problem
        cm = metrics.confusion_matrix(y_test, y_pred)
        print "Confusion matrix:"
        print cm
        
        # **********************************************************************
        # The mean_squared_error function computes the mean square error, 
        # which is a risk function corresponding to the expected value of the 
        # squared error loss or quadratic loss
        print "Mean square error: %.4f" % metrics.mean_squared_error(y_test, y_pred)
        
        # Accuracy classification score.
        print "Accuracy score: %.4f" % metrics.accuracy_score(y_test, y_pred)
        
        # **********************************************************************
        # Explained variance score: 1 is perfect prediction and 0 means that 
        # there is no linear relationship between X and Y.
        #print 'Variance score: %.4f' % classifier.score(X_test, y_test)
        
        # The r2_score function computes R^2, the coefficient of determination. 
        # It provides a measure of how well future samples are likely to be 
        # predicted by the model.
        print "The coefficient of determination: %.4f" % metrics.r2_score(y_test, y_pred)
        
        # **********************************************************************
        # Compute the recall. The recall is the ratio tp / (tp + fn) where tp 
        # is the number of true positives and fn the number of false negatives. 
        # The recall is intuitively the ability of the classifier to find all 
        # the positive samples.
        
        # Calculate metrics for each label, and find their unweighted mean. This 
        # does not take label imbalance into account.
        print "Recall(macro): %.4f" % metrics.recall_score(y_test, y_pred, average='macro')
        
        # Calculate metrics globally by counting the total true positives, false 
        # negatives and false positives.
        #print "Recall(micro): %.4f" % metrics.recall_score(y_test, y_pred, average='micro')
        
        # Calculate metrics for each label, and find their average, weighted by 
        # support (the number of true instances for each label). This alters 
        # 'macro' to account for label imbalance; it can result in an F-score 
        # that is not between precision and recall.
        #print "Recall(weighted): %.4f" % metrics.recall_score(y_test, y_pred, average='weighted')
        
        # If None, the scores for each class are returned.
        #print "Recall(None): ", metrics.recall_score(y_test, y_pred, average=None)
        
        # **********************************************************************
        # true positive (TP)
        # true negative (TN)
        # false positive (FP)
        # false negative (FN)
        # sensitivity or true positive rate (TPR)
        #     TPR = TP / P = TP / (TP + FN)
        # specificity (SPC) or True Negative Rate
        #     SPC = TN / N = TN / (FP + TN)
        
        # The function roc_curve computes the receiver operating characteristic 
        # curve, or ROC curve.
        # A receiver operating characteristic (ROC), or simply ROC curve, is a 
        # graphical plot which illustrates the performance of a binary classifier 
        # system as its discrimination threshold is varied. It is created by 
        # plotting the fraction of true positives out of the positives 
        # (TPR = true positive rate) vs. the fraction of false positives out of 
        # the negatives (FPR = false positive rate), at various threshold settings. 
        # TPR is also known as sensitivity, and FPR is one minus the specificity 
        # or true negative rate.
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        precision, recall, threshold = metrics.precision_recall_curve(y_test, y_pred)
        
        print "Sensitivity:"
        print tpr
        print "Specificity:"
        print fpr
        
        #print metrics.classification_report(y_test, y_pred)
        
        # **********************************************************************
        # Compute the F-beta score. The F-beta score is the weighted harmonic 
        # mean of precision and recall, reaching its optimal value at 1 and its 
        # worst value at 0. The beta parameter determines the weight of precision 
        # in the combined score. beta < 1 lends more weight to precision, while 
        # beta > 1 favors recall (beta -> 0 considers only precision, 
        # beta -> inf only recall).
        #print "F_{beta}(0.5): %.4f" % metrics.fbeta_score(y_test, y_pred, beta=0.5)
        
        # Compute precision, recall, F-measure and support for each class
        precision, recall, fbeta_score, support = \
            metrics.precision_recall_fscore_support(y_test, y_pred, \
                                                    beta=1, average="macro", pos_label=0)
        fbeta_score = float(0 if fbeta_score is None else fbeta_score)
        support = float(0 if support is None else support)
        print "[class=0] F_{beta}(1): %.4f" % fbeta_score
        print "[class=0] Support: %.4f" % support
        
        precision, recall, fbeta_score, support = \
            metrics.precision_recall_fscore_support(y_test, y_pred, \
                                                    beta=1, average="macro", pos_label=1)
        fbeta_score = float(0 if fbeta_score is None else fbeta_score)
        support = float(0 if support is None else support)
        print "[class=1] F_{beta}(1): %.4f" % fbeta_score
        print "[class=1] Support: %.4f" % support
