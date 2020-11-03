from src.helpers.get_rescaled_pred import get_rescaled_pred


def get_raw_with_prediction(model, dataset, device, index):
    raw_prediction, rescaled_pred = get_rescaled_pred(model, dataset, device, index)
    raw_data, raw_label = dataset.get_raw_item_with_label_filter(index)
    return raw_data, raw_label, raw_prediction
