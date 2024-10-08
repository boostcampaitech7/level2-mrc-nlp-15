import csv
from datasets import Dataset

train_dataset_path = "../data/train_dataset/train/dataset.arrow"
validation_dataset_path = "../data/train_dataset/validation/dataset.arrow"

dataset = Dataset.from_file(train_dataset_path)
val_dataset = Dataset.from_file(validation_dataset_path)
df = dataset.to_pandas()
v_df = val_dataset.to_pandas()

print(df.info())

df = df.drop(columns=['__index_level_0__'])
df = df.reset_index(drop=True)

# make new df based on the original df
new_csv = df['title']
new_csv = new_csv.to_csv('../data/train_dataset/train/title.csv', index=False, encoding='utf-32')

v_new_csv = v_df['title']
v_new_csv = v_new_csv.to_csv('../data/train_dataset/validation/title.csv', index=False, encoding='utf-32')

# Add new Column
new_csv = df['context']
new_csv = new_csv.to_csv('../data/train_dataset/train/context.csv', index=False, encoding='utf-32')

v_new_csv = v_df['context']
v_new_csv = v_new_csv.to_csv('../data/train_dataset/validation/context.csv', index=False, encoding='utf-32')

new_csv = df['question']
new_csv = new_csv.to_csv('../data/train_dataset/train/question.csv', index=False, encoding='utf-32')

v_new_csv = v_df['question']
v_new_csv = v_new_csv.to_csv('../data/train_dataset/validation/question.csv', index=False, encoding='utf-32')

new_csv = df['answers']
new_csv = new_csv.to_csv('../data/train_dataset/train/answers.csv', index=False, encoding='utf-32')

v_new_csv = v_df['answers']
v_new_csv = v_new_csv.to_csv('../data/train_dataset/validation/answers.csv', index=False, encoding='utf-32')

print(df.head()['question'])
print(df.head()['id'])
print(df.head()['answers'])