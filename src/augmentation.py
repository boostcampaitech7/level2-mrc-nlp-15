def synonym_augmentation(train_dataset):
    pass


def co_shift_augmentation(train_dataset):
    return train_dataset


def synonym_replacement(train_dataset):
    return train_dataset


def random_insertion(train_dataset):
    return train_dataset


def random_deletion(train_dataset):
    return train_dataset


def random_swap(train_dataset):
    return train_dataset


def shift_augmentation(train_dataset):
    return train_dataset


def augmentation(train_dataset, aug_list):
    if 'synonym_replacement' in aug_list:
        train_dataset = synonym_replacement(train_dataset)

    if 'random_insertion' in aug_list:
        train_dataset = random_insertion(train_dataset)

    if 'random_deletion' in aug_list:
        train_dataset = random_deletion(train_dataset)

    if 'random_swap' in aug_list:
        train_dataset = random_swap(train_dataset)

    if 'shift' in aug_list:
        train_dataset = shift_augmentation(train_dataset)
        
    return train_dataset