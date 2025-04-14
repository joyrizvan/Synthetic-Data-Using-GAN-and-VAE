import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
import random
class statistical_evaluation:
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Initialize the StatisticalEvaluation class.
        
        Args:
            real_data (pd.DataFrame): real_data data DataFrame.
            synthetic_data (pd.DataFrame): synthetic_data data DataFrame.
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data

    ### 1. Statistical Similarity Evaluation
    def compare_kde_distributions(self, feature):
        plt.figure(figsize=(8, 4))
        sns.kdeplot(self.real_data[feature], label="real_data", fill=True, alpha=0.5)
        sns.kdeplot(self.synthetic_data[feature], label="synthetic_data", fill=True, alpha=0.5)
        plt.title(f"Distribution Comparison for {feature}")
        plt.legend()
        plt.show()
    
    def get_random_color(self):
        return np.random.rand(3,)  # returns an RGB tuple with values between 0 and 1

    def compare_kde_distributions_color(self, feature):
        plt.figure(figsize=(8, 4))

        real_color = self.get_random_color()
        synth_color = self.get_random_color()

        sns.kdeplot(self.real_data[feature], label="real_data", fill=True, alpha=0.5, color=real_color)
        sns.kdeplot(self.synthetic_data[feature], label="synthetic_data", fill=True, alpha=0.5, color=synth_color)

        plt.title(f"Distribution Comparison for {feature}")
        plt.legend()
        plt.show()

    def compare_all_distributions(self):
        #self.synthetic_data = self.synthetic_data.drop('Churn', axis=1)
        numerical_cols = self.synthetic_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            self.compare_kde_distributions_color(col)

    def ks_test(self):
        # Define column names
        columns = self.real_data.select_dtypes(include=[np.number]).columns

        # Compute KS test results
        ks_results = []
        for col in columns:
            ks_stat, p_value = ks_2samp(self.real_data[col], self.synthetic_data[col])
            ks_results.append([col, ks_stat, p_value])

        # Create a DataFrame for displaying results
        ks_df = pd.DataFrame(ks_results, columns=["Feature", "KS Statistic", "P-Value"])

        # Format the table nicely
        ks_df["KS Statistic"] = ks_df["KS Statistic"].round(4)
        ks_df["P-Value"] = ks_df["P-Value"].apply(lambda x: f"{x:.4f}")


        return ks_df

    def chi_square_test(self):
        results = []  # Store results for plotting
        for col in self.real_data.select_dtypes(exclude=[np.number]).columns:
            contingency_table = pd.crosstab(self.real_data[col],  self.synthetic_data[col])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            print(f"{col}: Chi-Square Stat={chi2:.4f}, p-value={p:.4f}")
            results.append({'Feature': col, 'Chi2': chi2, 'P-Value': p})
        return pd.DataFrame(results)
    def get_random_palette_name(self):
        palette_options = [
            "Purples_r", "Purples", "Blues_r", "Blues", "Greens_r", "Greens",
            "Oranges_r", "Oranges", "Reds_r", "Reds", 
            "BuGn_r", "BuGn", "YlGnBu", "YlOrRd", "PuBuGn_r", "PuBu_r"
        ]
        return random.choice(palette_options)
    def bar_plot_test_results(self, results_df, significance_level=0.05):
        metric_name = results_df.columns[1]  # Dynamically get the second column name (e.g., "KS Statistic" or "Chi-Square Statistic")
        if metric_name == 'Wasserstein Distance':

                plt.figure(figsize=(10, 6))
                results_df = results_df.sort_values(by=metric_name, ascending=False)  # Sort for better visualization
                
                sns.barplot(x="Wasserstein Distance", y="Feature", data=results_df, palette=self.get_random_palette_name())
                
                plt.title("{} for Features".format(metric_name))
                plt.xlabel("{}".format(metric_name))
                plt.ylabel("Feature")
                plt.show()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Sort values for better visualization
            results_df = results_df.sort_values(by=metric_name, ascending=False)

            # Metric Bar Chart (KS Statistic or Chi-Square)
            sns.barplot(x=metric_name, y="Feature", data=results_df, ax=axes[0], palette=self.get_random_palette_name())
            axes[0].set_title(f"{metric_name} for Features")
            #axes[0].axvline(significance_level, color="red", linestyle="dashed", label="Significance Level")
            axes[0].legend()

            # P-Value Bar Chart
            results_df["Display P-Value"] = results_df["P-Value"].astype(float).apply(lambda x: max(x, 1e-4))

            # Sort for nice visualization
            results_df = results_df.sort_values(by="Display P-Value", ascending=True)

            # Plot the P-Value bar chart without log scale
            sns.barplot(x="Display P-Value", y="Feature", data=results_df, ax=axes[1], palette=self.get_random_palette_name())
            axes[1].set_title("P-Values for Features")
            axes[1].axvline(significance_level, color="red", linestyle="dashed", label="Significance Level")
            axes[1].legend()
            plt.tight_layout()
            plt.show()

    def plot_tsne(self):
        combined = pd.concat([self.real_data, self.synthetic_data], axis=0)
        labels = ['Real'] * len(self.real_data) + ['Synthetic'] * len(self.synthetic_data)

        # Adjust n_components to be at most the number of features
        n_components = min(10, combined.shape[1])  # Choose 10 or max available

        # Apply PCA
        pca = PCA(n_components=n_components).fit_transform(combined)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedded = tsne.fit_transform(pca)

        # Plot the result
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels, alpha=0.5)
        plt.title("t-SNE Visualization of Real vs. Synthetic Data")
        plt.show()


    def wasserstein_distance(self):
        # Compute Wasserstein Distance for each numerical column
        wasserstein_results = {}
        for col in self.real_data.select_dtypes(include=[np.number]).columns:
            distance = wasserstein_distance(self.real_data[col], self.synthetic_data[col])
            wasserstein_results[col] = distance
        wasserstein_df = pd.DataFrame(list(wasserstein_results.items()), columns=['Feature', 'Wasserstein Distance'])
        return wasserstein_df
