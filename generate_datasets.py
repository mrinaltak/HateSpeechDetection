from datasets import load_dataset

train_dataset = load_dataset('csv', data_files={'train': './OLID_dataset/olid-training-v1.0.tsv'}, delimiter="\t")
test_a_dataset = load_dataset('csv', data_files={'test': './OLID_dataset/olid_test_subtask_a.csv'}, delimiter="\t")
test_b_dataset = load_dataset('csv', data_files={'test': './OLID_dataset/olid_test_subtask_b.csv'}, delimiter="\t")
test_c_dataset = load_dataset('csv', data_files={'test': './OLID_dataset/olid_test_subtask_c.csv'}, delimiter="\t")
