# Income_Prediction
This project aims to predict the income of individuals based on their demographic and employment data using XGBoost and LightGBM models.

# Dataset

The dataset used in this project is the Adult dataset from the UCI Machine Learning Repository. It contains demographic and employment-related attributes of individuals, along with their income level (<=50K or >50K).

# Prerequisites

    Python 3.x
    pandas
    NumPy
    scikit-learn
    XGBoost
    LightGBM
    matplotlib
    
# Getting Started

    Clone this repository: git clone https://github.com/yourusername/adult-income-classification.git
    Navigate to the project directory: cd adult-income-classification
    Install the required packages: pip install -r requirements.txt
    Run the code: python adult_income_classification.py
    
 # Code Explanation

    Load the dataset using pandas.
    Remove missing values by replacing "?" with NaN and dropping NaN values.
    Perform outlier handling using IQR method.
    Label encode categorical columns using scikit-learn's LabelEncoder.
    Perform feature selection on categorical columns using scikit-learn's SelectKBest and chi2 methods.
    Combine the selected features with numerical columns.
    Split the data into training and test sets.
    Standardize the numerical features using scikit-learn's StandardScaler.
    Train XGBoost and LightGBM models and evaluate them using mean squared error (MSE).
    Visualize the comparison of the two models using matplotlib.
    
#License

This project is licensed under the MIT License.
