import kagglehub
import pandas
import os

# Download latest version
path = kagglehub.dataset_download("ramaqubra/fake-and-real-news-datasets")

print("Path to dataset files:", path)

fake_df = pandas.read_csv(os.path.join(path, 'Fake.csv'))
real_df = pandas.read_csv(os.path.join(path, 'True.csv'))
fake_headers = fake_df['title'].to_list()
real_headers = real_df['title'].to_list()
fake_articles = fake_df['text'].to_list()
real_articles = real_df['text'].to_list()