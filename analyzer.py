import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import transformers
import torch
import nltk
import warnings
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from nltk.corpus import stopwords

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class CollegeReviewAnalyzer:
    def __init__(self, df_path):
        """
        Initialize the dataset
        """
        self.df = pd.read_csv(df_path, lineterminator='\n')
        self.df_path = df_path
        # Set option to show full text in DataFrame columns
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', 0)           # Auto-detect width
        pd.set_option('display.max_colwidth', None) # Show full content in each cell
        # Initialize sentiment analysis tools
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def show_data_samples(self, n: int = 10):
        """
        Display sample data from the dataset.
        
        Parameters:
        - n: int (number of samples to display, default is 10)
        """
        return self.df.head(n)
    
    def get_value_counts(self, column_name: str):
        """
        Get value counts of the specified column for the dataset.
        
        Parameters:
        - column_name: str (name of the column to count values for)
        """
        return self.df[column_name].value_counts()
    
    def check_missing_values(self):
        """
        Check for missing values and data types in the dataset.
        """
        return self.df.isnull().sum()

    def plot_missing_values_heatmap(self):
        """
        Plot a heatmap visualizing missing values in the dataset.
        """
        plt.figure(figsize=(10, 4))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.show()
    
    def get_descriptive_statistics(self, column_name: str):
        """
        Get summary statistics for the specified column in the dataset.

        Parameters:
        - column_name: str (name of the column to get summary statistics for)
        """
        return self.df[column_name].describe()
    
    def plot_variable_distributions(self, column_names: list):
        """
        Plot distribution of specified variables (columns) in the dataset.
        The number of subplots will match the number of variables provided.
        
        Parameters:
        - column_names: list of str (names of columns to plot distributions for)
        """
        num_vars = len(column_names)
        plt.figure(figsize=(5 * num_vars, 5))
        for i, col in enumerate(column_names, 1):
            plt.subplot(1, num_vars, i)
            sns.histplot(self.df[col], bins=10, kde=False)
            plt.title(f'Distribution - {col}')
            plt.xlabel(col)
            plt.ylabel('Count')

        plt.tight_layout()
        plt.show()
    
    def analyze_sentence_lengths(self, column_name: str):
        """
        Analyze and plot the distribution of review lengths for the dataset using plot_variable_distributions.
        """
        # Calculate review lengths if not already present
        self.df[column_name] = self.df[column_name].apply(len)
        return self.df[column_name]

    def clean_text(self, text):
        """
        Clean the input text by removing newlines, special characters, digits, and punctuation symbols.
        """
        cleaned_text = str(text).replace('\n', ' ').replace('\r', ' ')
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_text)
        return cleaned_text

    def get_sentiment(self, review):
        """
        Perform sentiment analysis on the input review text.
        Returns: 'positive', 'negative', or 'neutral'.
        """
        inputs = self.tokenizer(review, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        positive_probability = probabilities[:, 1].item()
        negative_probability = probabilities[:, 0].item()
        if positive_probability > 0.53:
            return "positive"
        elif negative_probability > 0.53:
            return "negative"
        else:
            return "neutral"

    def add_sentiment_columns(self, review_column='review', cleaned_column='cleaned_reviews', sentiment_column='sentiment', save_path=None):
        """
        Clean the reviews and add sentiment analysis columns to the DataFrame.
        Removes rows where cleaned review starts with 'var allowPaste'.
        Optionally saves the DataFrame with sentiments to a CSV file if save_path is provided.
        """
        self.df[cleaned_column] = self.df[review_column].apply(self.clean_text)
        filter_condition = self.df[cleaned_column].apply(lambda x: not str(x).startswith("var allowPaste"))
        self.df = self.df[filter_condition].copy()
        self.df[sentiment_column] = self.df[cleaned_column].apply(self.get_sentiment)
        if save_path:
            self.save_with_sentiments(save_path, columns=[review_column, cleaned_column, sentiment_column])
        return self.df[[review_column, cleaned_column, sentiment_column]]

    def save_with_sentiments(self, output_path, columns=None):
        """
        Save the DataFrame (optionally with selected columns) to a CSV file at output_path.
        """
        if columns:
            self.df.to_csv(output_path, index=False, columns=columns)
        else:
            self.df.to_csv(output_path, index=False)

    def show_sentiment_counts(self, sentiment_column='sentiment'):
        """
        Print value counts for the sentiment column.
        """
        return self.df[sentiment_column].value_counts()

    def show_dtypes(self):
        """
        Print the data types of the DataFrame columns.
        """
        return self.df.dtypes

    def show_head(self, n: int = 35):
        """
        Print the first n rows of the DataFrame.
        """
        return self.df.head(n)


def main():
    # Initialize the analyzer with data paths
    analyzer = CollegeReviewAnalyzer(
        df_path='datasets/collegereview2021.csv',
    )
    print(f"\nInitialized dataset path: {analyzer.df_path}")

    print("\nShowing data samples:")
    print(analyzer.show_data_samples())

    # Use a variable for the column name
    column_name = 'college'
    print(f"\nValue counts for variable: {column_name}")
    print(analyzer.get_value_counts(column_name))
    
    print("\nChecking for missing values:")
    print(analyzer.check_missing_values())
    
    print("\nPlotting missing values heatmap:")
    analyzer.plot_missing_values_heatmap()
    
    column_name = 'rating'
    print(f"\nRating statistics for 2021: {column_name}")
    print(analyzer.get_descriptive_statistics(column_name))
    
    column_name = 'rating'
    print("\nPlotting distributions for: {column_name}")
    analyzer.plot_variable_distributions([column_name])
    
    column_name = 'review'
    print(f"\nAnalyzing lengths for: {column_name}")
    print(analyzer.analyze_sentence_lengths(column_name))

    # Sentiment analysis and further analysis
    print("\nAdding sentiment columns and cleaning reviews...")
    analyzer.add_sentiment_columns(review_column='review', cleaned_column='cleaned_reviews', sentiment_column='sentiment', save_path='output_with_sentiments.csv')
    
    print("\nFirst 5 rows with sentiment:")
    print(analyzer.show_head())
    
    print("\nSentiment value counts:")
    print(analyzer.show_sentiment_counts('sentiment'))
    
    print("\nData types of DataFrame columns:")
    print(analyzer.show_dtypes())

if __name__ == "__main__":
    main()