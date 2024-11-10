from sklearn import metrics
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix

def calc_acc(y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	return accuracy

# Precision, Recall, F-score
def calc_prf(y_true, y_pred, pos_label=1, is_binary = True):
	#beta: The strength of recall versus precision in the F-score.
	average_types = ["micro", "macro", "weighted"]
	if is_binary:
		average_types += ["binary"]
	
	perf_dict = {}
	for average_type in average_types:
		precision, recall, f1, support = precision_recall_fscore_support(
			y_true, y_pred, 
			beta = 1.0, 
			pos_label=pos_label, 
			average=average_type
		)
		f_0_5 = fbeta_score(y_true, y_pred, average = average_type, beta=0.5)
		f2 = fbeta_score(y_true, y_pred, average = average_type, beta=2.0)
		perf_dict[average_type] = {
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"f0.5": f_0_5,
			"f2": f2,
		}
	return perf_dict