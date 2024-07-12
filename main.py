import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

train_data = pd.read_csv('twitter_training.csv')
validation_data = pd.read_csv('twitter_validation.csv')

print(train_data.head())
print(validation_data.head())

nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if isinstance(text, str):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    return 0.0

train_data['sentiment'] = train_data.iloc[:, -1].apply(get_sentiment)
validation_data['sentiment'] = validation_data.iloc[:, -1].apply(get_sentiment)

train_data.head()
validation_data.head()

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.histplot(train_data['sentiment'], bins=50, kde=True)
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(validation_data['sentiment'], bins=50, kde=True)
plt.title('Sentiment Distribution in Validation Data')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=[train_data['sentiment'], validation_data['sentiment']], palette="Set2")
plt.title('Boxplot of Sentiment Scores')
plt.xlabel('Dataset')
plt.ylabel('Sentiment Score')
plt.xticks([0, 1], ['Training Data', 'Validation Data'])
plt.show()

brand=validation_data['Facebook']
sentiment=validation_data['Irrelevant']
print(brand,sentiment)

brand.dropna()
sentiment.dropna()

brand.value_counts().plot(kind='bar');
plt.show()
sentiment.value_counts().plot(kind='bar');
plt.show()

train_mean = train_data['sentiment'].mean()
validation_mean = validation_data['sentiment'].mean()

print(f"Mean Sentiment Score in Training Data: {train_mean}")
print(f"Mean Sentiment Score in Validation Data: {validation_mean}")

train_median = train_data['sentiment'].median()
validation_median = validation_data['sentiment'].median()

print(f"Median Sentiment Score in Training Data: {train_median}")
print(f"Median Sentiment Score in Validation Data: {validation_median}")


