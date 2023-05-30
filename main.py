# Import necessary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Function to plot Lasso coefficients for different alpha values
def plot_lasso():
    """Plots Lasso coefficients as a function of the regularization."""

    # Setting plot style and context
    sns.set_style('white')
    sns.set_context("poster",)
    
    # Creating a subplot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
    
    titles = ['Overall Fractal Dimension', 'Veins', 'Arteries', 'Ratio']

    # Read data
    X1, y1, _, _ = read_data_reg(target='D', dropna=False, norm=True, remove_collinear=True, subset=None)
    X2, y2, _, _ = read_data_reg(target='Art_D_qn', dropna=False, norm=True, remove_collinear=True, subset=None)
    X3, y3, _, _ = read_data_reg(target='Vein_D_qn', dropna=False, norm=True, remove_collinear=True, subset=None)
    X4, y4, _, _ = read_data_reg(target='AVR_D_qn', dropna=False, norm=True, remove_collinear=True, subset=None)

    # List of dataframes
    X_list = [X1, X2, X3, X4]
    y_list = [y1, y2, y3, y4]

    # Dictionary to rename columns
    rename_dict = { 
        # Your dictionary goes here
    }
    
    # Renaming columns
    for i in range(4):
        X_list[i].columns = [rename_dict[name] if name in rename_dict else name for name in X_list[i].columns]

    # Creating color dictionary
    feature_names_set = set(X1.columns).union(set(X2.columns)).union(set(X3.columns)).union(set(X4.columns))
    colors_dict = dict(zip(feature_names_set, sns.color_palette('hsv', len(feature_names_set))))

    # Line styles
    linestyles = ['-', '--', '-.', ':']

    # Looping through all axes
    for i, ax in enumerate(axes.flatten()):
        X = X_list[i]
        y = y_list[i]
        ax.set_title(titles[i])

        # Compute Lasso coefficients for different alpha values
        coefs = []
        alphas = np.logspace(-4, 1, 100)
        feature_names = X.columns
        for a in alphas:
            lasso = Lasso(alpha=a)
            lasso.fit(X, y)
            coefs.append(lasso.coef_)
        
        # Compute mean of coefficients
        coefs = np.array(coefs)
        mean_coefs = np.mean(np.abs(np.array(coefs)), axis=0)
        sorted_indices = np.argsort(mean_coefs)[::-1]
        sorted_feature_names = np.array(feature_names)[sorted_indices].tolist()

        # Select top 15 features
        top_features = sorted_feature_names[:15]

        # Plot coefficients
        for coef, feature in zip(coefs.T[sorted_indices], sorted_feature_names):
            line, = ax.plot.)
                if feature in top_features:
                  linestyle = linestyles[top_features.index(feature) % len(linestyles)]
                  line, = ax.plot(alphas, coef, label=feature, linestyle=linestyle)
                  line.set_color(colors_dict[feature])

          ax.set_xscale('log')
          ax.legend()

      # Save the plot
      plt.savefig("lasso_path.png")
      plt.close()

  # Call the function
  if __name__ == "__main__":
      plot_lasso()
