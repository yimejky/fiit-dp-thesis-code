from operator import itemgetter


def get_dataset_info(dataset, dataloaders_obj):
    train_dataset, valid_dataset, test_dataset = itemgetter(
        'train_dataset', 'valid_dataset', 'test_dataset')(dataloaders_obj)
    train_size, valid_size, test_size = itemgetter(
        'train_size', 'valid_size', 'test_size')(dataloaders_obj)

    print(f'train {train_size}, valid_size {valid_size}, test {test_size}, full {len(dataset)}')
    print(f'train indeces', str(sorted(train_dataset.indices)))
    print(f'valid indeces', str(sorted(valid_dataset.indices)))
    print(f'test indeces', str(sorted(test_dataset.indices)))
