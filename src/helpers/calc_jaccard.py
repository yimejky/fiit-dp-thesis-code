
def calc_jaccard(y_true, y_pred, smooth=1e-6):
    intersection = (y_true * y_pred).sum()
    jaccard = (intersection + smooth) / (y_true.sum() + y_pred.sum() - intersection + smooth)
    return jaccard
