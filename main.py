import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

# Load Data
df = pd.read_excel("data/asean_financial_depth.xlsx")

# Clean country strings and sort
df["COUN"] = df["COUN"].astype(str).str.strip()
df = df.sort_values(["COUN", "YEAR"]).reset_index(drop=True)

# Differencing for stationarity
df["dM2"] = df.groupby("COUN")["M2"].diff()
df["dPRVT"] = df.groupby("COUN")["PRVT"].diff()

df_clean = df.dropna(subset=["M2", "PRVT"]).copy()
df_diff = df.dropna(subset=["dM2", "dPRVT"]).copy()

# Overall ASEAN Correlation
print(f"ASEAN Level Correlation: {df_clean['M2'].corr(df_clean['PRVT']):.3f}")

# Indonesia Correlation
df_idn = df[df["COUN"] == "Indonesia"].sort_values("YEAR").copy()
df_idn_diff = df_diff[df_diff["COUN"] == "Indonesia"].sort_values("YEAR").copy()

idn_corr = df_idn["M2"].corr(df_idn["PRVT"])
print(f"Indonesia Level Correlation: {idn_corr:.3f}")

# ADF Stationarity Tests
print("\nStationarity Tests (ADF):")
print(f"M2 Level p-value: {adfuller(df_idn['M2'].dropna(), result_object=False)[1]:.4f}")
print(f"PRVT Level p-value: {adfuller(df_idn['PRVT'].dropna(), result_object=False)[1]:.4f}")
print(f"dM2 Diff p-value: {adfuller(df_idn_diff['dM2'].dropna(), result_object=False)[1]:.4f}")
print(f"dPRVT Diff p-value: {adfuller(df_idn_diff['dPRVT'].dropna(), result_object=False)[1]:.4f}")

# Simple OLS
X_idn = sm.add_constant(df_idn_diff["dM2"].dropna())
y_idn = df_idn_diff["dPRVT"].dropna()
model_idn_ols = sm.OLS(y_idn, X_idn).fit()

print("\nIndonesia OLS (dM2 -> dPRVT):")
print(model_idn_ols.summary())

# Granger Causality
var_data = df_idn_diff[["dPRVT", "dM2"]].dropna()

gc_fwd = grangercausalitytests(var_data[["dPRVT", "dM2"]], maxlag=1)
gc_rev = grangercausalitytests(var_data[["dM2", "dPRVT"]], maxlag=1)

p_fwd = gc_fwd[1][0]['ssr_ftest'][1]
p_rev = gc_rev[1][0]['ssr_ftest'][1]

print("\nGranger Causality:")
print(f"Forward (dM2 -> dPRVT) p-value: {p_fwd:.4f}")
print(f"Reverse (dPRVT -> dM2) p-value: {p_rev:.4f}")

# Comparing L1, L2 and L3
print("\n--- LAG SELECTION & COMPARISON ---")

model_select = VAR(var_data)
select_results = model_select.select_order(maxlags=3)
print(select_results.summary())

print("\nGranger Causality across Lags (dM2 -> dPRVT):")
for lag in [1, 2, 3]:
    try:
        gc_lag = grangercausalitytests(var_data[["dPRVT", "dM2"]], maxlag=lag)
        p_val = gc_lag[lag][0]['ssr_ftest'][1]
        print(f"Lag {lag} p-value: {p_val:.4f}")
    except Exception as e:
        print(f"Lag {lag}: Not enough observations to calculate.")

# VAR Model
var_model = VAR(var_data)
var_res = var_model.fit(maxlags=1)

print("\nVAR Model Summary:")
print(var_res.summary())

# FEVD Calculation
def calculate_manual_fevd(var_result, periods=5):
    A1 = var_result.params.iloc[1:].values.T
    Sigma = var_result.sigma_u
    k = A1.shape[0]
    
    P = np.linalg.cholesky(Sigma)
    
    Phi = [np.eye(k)]
    for h in range(1, periods):
        Phi.append(Phi[-1] @ A1)
        
    Theta = [Phi_h @ P for Phi_h in Phi]
    
    fevd_dict = {var_name: np.zeros((periods, k)) for var_name in var_result.names}
    
    for eq_idx, eq_name in enumerate(var_result.names):
        mse_accum = np.zeros(k)
        for h in range(periods):
            step_contrib = Theta[h][eq_idx, :] ** 2
            mse_accum += step_contrib
            total_mse = np.sum(mse_accum)
            fevd_dict[eq_name][h, :] = mse_accum / total_mse
            
    return fevd_dict

# Run manual FEVD
fevd_results = calculate_manual_fevd(var_res, periods=5)

df_fevd_prvt = pd.DataFrame(fevd_results['dPRVT'], columns=['dPRVT', 'dM2'])
df_fevd_m2 = pd.DataFrame(fevd_results['dM2'], columns=['dPRVT', 'dM2'])

print("\nFEVD Summary Table (dPRVT):")
print(df_fevd_prvt)

print("\nFEVD Summary Table (dM2):")
print(df_fevd_m2)

# Extract 1-Year Horizon (Index 1)
m2_to_prvt = df_fevd_prvt.loc[1, 'dM2'] * 100
prvt_to_m2 = df_fevd_m2.loc[1, 'dPRVT'] * 100

print(f"\nVariance Analysis at Year 1 (FEVD):")
print(f"M2 -> Credit: {m2_to_prvt:.1f}%")
print(f"Credit -> M2: {prvt_to_m2:.1f}%")

# IRF Visualization
irf = var_res.irf(periods=5)
fig = irf.plot(orth=True)
plt.suptitle("Indonesia: Impulse Response Functions (95% Bands)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("fig3_indonesia_irf.png", dpi=300)
plt.close()

print("\nExecution complete.")