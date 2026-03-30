from sklearn.metrics import f1_score, precision_score, recall_score


def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def compute_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def compute_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)