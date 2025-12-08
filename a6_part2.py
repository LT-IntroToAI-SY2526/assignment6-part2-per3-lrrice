"""
Assignment 6 Part 2: House Price Prediction (Multivariable Regression)

This assignment predicts house prices using MULTIPLE features.
Complete all the functions below following the in-class car price example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the house price data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    data = pd.read_csv(filename)
    
    print("=== Car Price Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    print(f"\nBasic statistics:")
    print(data.describe())
    
    print(f"\nColumn names: {list(data.columns)}")
    
    return data


def visualize_features(data):
    """
    Create 4 scatter plots (one for each feature vs Price)
    
    Args:
        data: pandas DataFrame with features and Price
    """
    # TODO: Create a figure with 2x2 subplots, size (12, 10)
    
    # TODO: Add a main title: 'House Features vs Price'
    
    # TODO: Plot 1 (top left): SquareFeet vs Price
    #       - scatter plot, color='blue', alpha=0.6
    #       - labels and title
    #       - grid
    
    # TODO: Plot 2 (top right): Bedrooms vs Price
    #       - scatter plot, color='green', alpha=0.6
    #       - labels and title
    #       - grid
    
    # TODO: Plot 3 (bottom left): Bathrooms vs Price
    #       - scatter plot, color='red', alpha=0.6
    #       - labels and title
    #       - grid
    
    # TODO: Plot 4 (bottom right): Age vs Price
    #       - scatter plot, color='orange', alpha=0.6
    #       - labels and title
    #       - grid
    
    # TODO: Use plt.tight_layout() to make plots fit nicely
    
    # TODO: Save the figure as 'feature_plots.png' with dpi=300
    
    # TODO: Show the plot
    pass


def prepare_features(data):
    """
    Separate features (X) from target (y)
    
    Args:
        data: pandas DataFrame with all columns
    
    Returns:
        X - DataFrame with feature columns
        y - Series with target column
    """
    feature_columns = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'age']
    X = data[feature_columns]
    y = data['Price']
    
    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    
    return X, y


def split_data(X, y):
    """
    Split data into training and testing sets
    
    Args:
        X: features DataFrame
        y: target Series
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    print(f"\n=== Data Split (Matching Unplugged Activity) ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train a multivariable linear regression model
    
    Args:
        X_train: training features (scaled)
        y_train: training target values
        feature_names: list of feature column names
    
    Returns:
        trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    
    print(f"\nEquation:")
    equation = f"Price = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model's performance
    
    Args:
        model: trained model
        X_test: testing features (scaled)
        y_test: testing target values
        feature_names: list of feature names
    
    Returns:
        predictions array
    """
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")
    
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    
    return predictions


def compare_predictions(y_test, predictions, num_examples=5):
    """
    Show side-by-side comparison of actual vs predicted prices
    
    Args:
        y_test: actual prices
        predictions: predicted prices
        num_examples: number of examples to show
    """
    print(f"\n=== Prediction Examples ===")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)
    
    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100
        
        print(f"${actual:>13.2f}   ${predicted:>13.2f}   ${error:>10.2f}   {pct_error:>6.2f}%")


def make_prediction(model, sqft, bedrooms, bathrooms, age):
    """
    Make a prediction for a specific house
    
    Args:
        model: trained LinearRegression model
        sqft: square footage
        bedrooms: number of bedrooms
        bathrooms: number of bathrooms
        age: age of house in years
    
    Returns:
        predicted price
    """
    car_features = pd.DataFrame([[bedrooms, age, bathrooms, sqft]], 
                                 columns=['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age'])
    predicted_price = model.predict(car_features)[0]
    
    brand_name = ['Toyota', 'Honda', 'Ford'][bedrooms]
    
    print(f"\n=== New Prediction ===")
    print(f"Car specs: {bathrooms:.0f}k miles, {age} years old, {brand_name}")
    print(f"Predicted price: ${predicted_price:,.2f}")
    
    return predicted_price


if __name__ == "__main__":
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)
    
    # Step 1: Load and explore
    # TODO: Call load_and_explore_data() with 'house_prices.csv'
    data = load_and_explore_data('house_prices.csv')
    # Step 2: Visualize features
    # TODO: Call visualize_features() with the data
    
    # Step 3: Prepare features
    # TODO: Call prepare_features() and store X and y
    X, y = prepare_features(data)
    # Step 4: Split data
    # TODO: Call split_data() and store X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Step 5: Train model
    # TODO: Call train_model() with training data and feature names (X.columns)
    
    # Step 6: Evaluate model
    # TODO: Call evaluate_model() with model, test data, and feature names
    
    # Step 7: Compare predictions
    # TODO: Call compare_predictions() showing first 10 examples
    
    # Step 8: Make a new prediction
    # TODO: Call make_prediction() for a house of your choice
    
    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part2_writeup.md!")
    print("=" * 70)