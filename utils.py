from datasets import load_dataset

def load_data(path = './OLID_dataset'):
    train_dataset = load_dataset(path = './OLID_dataset', data_files={'train': 'olid_train_v2.csv'}, delimiter="\t", cache_dir='./olid_cache')
    val_dataset = load_dataset(path = './OLID_dataset', data_files={'val': 'olid_val_v2.csv'}, delimiter="\t", cache_dir='./olid_cache')
    test_a_dataset = load_dataset(path = './OLID_dataset', data_files={'test': 'olid_test_subtask_a.csv'}, delimiter="\t", cache_dir='./olid_cache')
    test_b_dataset = load_dataset(path = './OLID_dataset', data_files={'test': 'olid_test_subtask_b.csv'}, delimiter="\t", cache_dir='./olid_cache')
    test_c_dataset = load_dataset(path = './OLID_dataset', data_files={'test': 'olid_test_subtask_c.csv'}, delimiter="\t", cache_dir='./olid_cache')
    return train_dataset, val_dataset, test_a_dataset, test_b_dataset, test_c_dataset

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)