import kagglehub
import pandas
import os

# Download latest version
path = kagglehub.dataset_download("shanegerami/ai-vs-human-text")

df = pandas.read_csv(os.path.join(path, 'AI_Human.csv'))
outputs = df['generated'].to_list()[:5000]
texts = df['text'].to_list()[:5000]
print(outputs)

print("Path to dataset files:", path)