from operator import itemgetter


def get_dataset_info(dataset, split_dataset_obj):
    train_dataset, valid_dataset, test_dataset = itemgetter('train_dataset', 'valid_dataset', 'test_dataset')(
        split_dataset_obj)
    train_size, valid_size, test_size = itemgetter('train_size', 'valid_size', 'test_size')(split_dataset_obj)

    print(f'train {train_size}, valid_size {valid_size}, test {test_size}, full {len(dataset)}')
    print(f'train indices', str(sorted(train_dataset.indices)))
    print(f'valid indices', str(sorted(valid_dataset.indices)))
    print(f'test indices', str(sorted(test_dataset.indices)))
