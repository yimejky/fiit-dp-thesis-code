from operator import itemgetter


def get_datasets_info(datasets_obj):
    train_dataset, valid_dataset, test_dataset = itemgetter('train_dataset', 'valid_dataset', 'test_dataset')(datasets_obj)
    train_size, valid_size, test_size, dataset = itemgetter('train_size', 'valid_size', 'test_size', 'dataset')(datasets_obj)

    print(f'train {train_size}, valid_size {valid_size}, test {test_size}, full {len(dataset)}')
    print(f'train indeces', str(sorted(train_dataset.indices)))
    print(f'valid indeces', str(sorted(valid_dataset.indices)))
    print(f'test indeces', str(sorted(test_dataset.indices)))
