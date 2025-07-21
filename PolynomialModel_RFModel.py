# --------- Code 3A: Polynomial Curve Fit (W, Tm, Tm2) to Tc ---------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heat_data.csv")

X_poly_features = df[['W', 'Tm', 'Tm2']].values
y_tc = df['Tc'].values

poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X_poly_features)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_tc)

terms = poly.get_feature_names_out(['W', 'Tm', 'Tm2'])
coeffs = poly_model.coef_
intercept = poly_model.intercept_

print("--- Polynomial Fit Equation ---")
equation = f"Tc = {intercept:.4f}"
for coef, term in zip(coeffs, terms):
    equation += f" + ({coef:.4f})*{term}"
print(equation)

print(f"Polynomial R² score: {poly_model.score(X_poly, y_tc):.4f}")

# Polynomial prediction curve
W_plot = np.linspace(df['W'].min(), df['W'].max(), 200)
tm_lookup = df.groupby('W')[['Tm', 'Tm2']].mean().reindex(W_plot, method='nearest').values
X_plot_poly = np.column_stack((W_plot, tm_lookup[:, 0], tm_lookup[:, 1]))
X_plot_poly_transformed = poly.transform(X_plot_poly)
Tc_poly_pred = poly_model.predict(X_plot_poly_transformed)

plt.figure(figsize=(8, 5))
plt.scatter(df['W'], df['Tc'], alpha=0.3, label='Simulation Data')
plt.plot(W_plot, Tc_poly_pred, color='black', linestyle='--', label='Polynomial Fit')
plt.xlabel("W (mm)")
plt.ylabel("Tc (°C)")
plt.title("Tc vs W (Polynomial Fit)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --------- Code 3B: Random Forest Prediction using W, Tm, Tm2 ---------
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_rf = df[['W', 'Tm', 'Tm2']]
y_rf = df['Tc']

X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"\nRandom Forest R² score on test set: {r2:.4f}")

# --------- Visualization ---------
W_plot = np.linspace(df['W'].min(), df['W'].max(), 200)
tm_lookup = df.groupby('W')[['Tm', 'Tm2']].mean().reindex(W_plot, method='nearest').values
X_vis = np.column_stack((W_plot, tm_lookup[:, 0], tm_lookup[:, 1]))
Tc_rf_pred = rf_model.predict(X_vis)

plt.figure(figsize=(8, 5))
plt.scatter(df['W'], df['Tc'], alpha=0.3, label='Data')
plt.plot(W_plot, Tc_rf_pred, color='red', label='RF Prediction using interpolated Tm & Tm2')
plt.xlabel("W (mm)")
plt.ylabel("Tc (°C)")
plt.title("Tc vs W (Random Forest using W, Tm, Tm2)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_pred, y_test, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Ideal Fit')
plt.xlabel("Predicted Tc")
plt.ylabel("Actual Tc")
plt.title("Predicted vs Actual Tc (Random Forest)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
