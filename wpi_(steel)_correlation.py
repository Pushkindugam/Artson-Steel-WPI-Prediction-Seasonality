import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    try:
        corr = df.select_dtypes('number').corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap – WPI Master Dataset")
        return fig
    except Exception as e:
        print("Error in correlation plot:", e)
        return None
