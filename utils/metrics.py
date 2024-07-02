

# True Positive
def true_positive(ground_truth, prediction):
    tp = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 1 and pred == 1:
            tp += 1
    return tp


# True Negative
def true_negative(ground_truth, prediction):
    tn = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 0 and pred == 0:
            tn += 1
    return tn

# False Positive


def false_positive(ground_truth, prediction):
    fp = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 0 and pred == 1:
            fp += 1
    return fp

# False Negative


def false_negative(ground_truth, prediction):
    fn = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 1 and pred == 0:
            fn += 1
    return fn


def accuracy(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)
    fp = false_positive(ground_truth, prediction)
    fn = false_negative(ground_truth, prediction)
    tn = true_negative(ground_truth, prediction)
    acc_score = (tp + tn) / (tp + tn + fp + fn)
    return acc_score


def precision(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)
    fp = false_positive(ground_truth, prediction)
    return tp / (tp + fp)


def recall(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)
    fn = false_negative(ground_truth, prediction)
    return tp / (tp + fn)


def f1(ground_truth, prediction):
    p = precision(ground_truth, prediction)
    r = recall(ground_truth, prediction)
    return 2 * p * r / (p + r)
