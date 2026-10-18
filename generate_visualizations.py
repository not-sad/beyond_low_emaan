# Result Visualization Script for Islamic AI Blind-spots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comprehensive_breakdown(results_csv_path, output_image_path):
    df = pd.read_csv(results_csv_path)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: User Preference
    sns.barplot(ax=axes[0, 0], data=df, x="guidance_preference", y="user_preference", hue="condition", errorbar=None, palette="muted")
    axes[0, 0].set_title("User Preference Alignment (Higher = Better)")
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    # Panel B: Non-dismissiveness
    sns.barplot(ax=axes[0, 1], data=df, x="category", y="non_dismissiveness", hue="condition", errorbar=None, palette="pastel")
    axes[0, 1].set_title("Non-Dismissiveness Scores by Category")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Panel C: Practical Usefulness
    sns.boxplot(ax=axes[1, 0], data=df, x="condition", y="practical_usefulness", palette="Set2")
    axes[1, 0].set_title("Distribution of Practical Usefulness Scores")
    
    # Panel D: Emotional Understanding
    sns.barplot(ax=axes[1, 1], data=df, x="category", y="emotional_understanding", hue="condition", errorbar=None, palette="deep")
    axes[1, 1].set_title("Emotional Understanding Score by Topic")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    plt.close()
