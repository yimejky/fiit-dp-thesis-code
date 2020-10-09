
def calc_dsc(y_true, y_pred, smooth=1e-6):
    intersection = (y_true * y_pred).sum()
    dsc = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
    return dsc


if __name__ == "__main__":
    import torch

    test_size = 4
    test_true = torch.zeros([test_size, test_size, test_size], dtype=torch.int32)
    test_true[0, 0, [0, 1, 2, 3]] = 1
    test_pred = torch.zeros([test_size, test_size, test_size], dtype=torch.float32)

    test_dsc = calc_dsc(test_true, test_pred)
    print(str(test_dsc))





