import logging


def calc_dsc(y_true, y_pred, smooth=1e-6):
    logging.debug(f'calc_dsc_helper1 y_true {y_true.shape}, y_pred {y_pred.shape}')
    assert y_true.size() == y_pred.size()

    y_true = y_true.contiguous().view(-1)
    y_pred = y_pred.contiguous().view(-1)

    intersection = (y_true * y_pred).sum()
    upper = 2. * intersection + smooth
    lower = y_true.sum() + y_pred.sum() + smooth
    dsc = upper / lower

    logging.debug(f'calc_dsc_helper2 smooth {smooth}, lower {lower}, upper {upper}, dsc {dsc}')

    return dsc


if __name__ == "__main__":
    import torch
    logging.basicConfig(level=logging.DEBUG)

    test_size = 4
    test_batch_size = 2

    test_true = torch.zeros([test_batch_size, test_size, test_size, test_size], dtype=torch.int32)
    test_true[[0, 1], 0, 0, 0] = 1
    test_true[[0], 0, 0, 1] = 1
    test_pred = torch.zeros([test_batch_size, test_size, test_size, test_size], dtype=torch.float32)
    test_pred[[0, 1], 0, 0, 0] = 1

    test_dsc = calc_dsc(test_true, test_pred)
    print(str(test_dsc))
