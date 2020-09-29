
def calc_dsc(y_true, y_pred, smooth=1.0):
    intersection = (y_true * y_pred).sum()
    dsc = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
    return dsc

def calc_jaccard(y_true, y_pred, smooth=1.0):
    intersection = (y_true * y_pred).sum()
    jaccard = (intersection + smooth) / (y_true.sum() + y_pred.sum() - intersection + smooth)
    return jaccard
