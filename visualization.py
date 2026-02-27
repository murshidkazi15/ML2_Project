import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(filepath):
    """Load the dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def plot_distributions(df):
    """Plot distributions of numerical features."""
    numerical_cols = ['Age', 'Family_Income', 'Study_Hours_per_Day', 'Attendance_Rate', 
                      'Stress_Index', 'CGPA']
    
    # Filter only columns that exist
    cols_to_plot = [col for col in numerical_cols if col in df.columns]
    
    if not cols_to_plot:
        return

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data=df, x=col, kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
    plt.savefig('numerical_distributions.png')
    plt.close()
    print("Saved numerical distributions plot.")

def plot_categorical_vs_target(df, target='Dropout'):
    """Plot categorical features against the target variable."""
    if target not in df.columns:
        return

    categorical_cols = ['Gender', 'Internet_Access', 'Part_Time_Job', 'Scholarship', 'Department']
    cols_to_plot = [col for col in categorical_cols if col in df.columns]
    
    if not cols_to_plot:
        return

    plt.figure(figsize=(15, 12))
    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot(2, 3, i)
        sns.countplot(data=df, x=col, hue=target)
        plt.title(f'{col} vs {target}')
        plt.xticks(rotation=45)
        plt.tight_layout()
    plt.savefig('categorical_vs_dropout.png')
    plt.close()
    print("Saved categorical vs target plot.")

def plot_correlation_matrix(df):
    """Plot correlation matrix of numerical features."""
    numerical_df = df.select_dtypes(include=[np.number])
    if numerical_df.empty:
        return
        
    plt.figure(figsize=(12, 10))
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    print("Saved correlation matrix plot.")

def main():
    filepath = 'student_dropout_dataset_v3.csv'
    df = load_data(filepath)
    
    if df is not None:
        print("Generating visualizations...")
        
        # Set visualization style
        sns.set_theme(style="whitegrid")
        
        plot_distributions(df)
        plot_categorical_vs_target(df)
        plot_correlation_matrix(df)
        
        print("All visualizations have been generated and saved as PNG files.")

if __name__ == "__main__":
    main()
