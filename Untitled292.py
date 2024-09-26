#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Data Preparation
assets = ['CSCO', 'ADBE', 'TGT', 'BBY', 'CVX', 'HES', 'T', 'CVS', 'WBA', 'BLK']
start_date = '2014-07-01'
end_date = '2024-07-01'

# Download historical data
data = yf.download(assets, start=start_date, end=end_date, interval='1wk')['Adj Close']

# Calculating weekly returns
returns = data.pct_change().dropna()

# Total number of assets
m = len(assets)
theta = 1 / (m + 1)

# Safe Logarithm Function
def safe_log(x):
    return np.log(np.where(x > 0, x, np.nan))

# Portfolio value at time T using equation (1)
log_portfolio_value = safe_log(1 + returns.mul(theta).sum(axis=1)).cumsum()

# Buy and Hold portfolio value using equation (2)
buy_and_hold_value = (data / data.iloc[0]).mul(theta).sum(axis=1)
log_buy_and_hold_value = safe_log(buy_and_hold_value / buy_and_hold_value.iloc[0])  # Normalizing the initial value 

# Calculation of Market Portfolio Value
initial_total = data.iloc[0].sum()
market_portfolio_value = data.sum(axis=1) / initial_total
log_market_portfolio_value = safe_log(market_portfolio_value)

# Plotting the results for portfolio values
plt.figure(figsize=(14, 7))
plt.plot(log_portfolio_value, label='Log Portfolio Value (Rebalanced)')
plt.plot(log_buy_and_hold_value, label='Log Buy and Hold Portfolio Value')
plt.plot(log_market_portfolio_value, label='Log Market Portfolio Value')
plt.legend()
plt.title('Log Portfolio Values over Time')
plt.xlabel('Time')
plt.ylabel('Log Value')
plt.grid(True)
plt.show()

# Calculation of Maximal Drawdown (MDD)
def calculate_mdd(series):
    # Convert numpy array to pandas Series if it's not already one
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    peak = series.cummax()
    drawdown = (series - peak) / peak
    return drawdown.min()

log_mdd = calculate_mdd(np.exp(log_portfolio_value))
buy_and_hold_mdd = calculate_mdd(np.exp(log_buy_and_hold_value))
market_mdd = calculate_mdd(np.exp(log_market_portfolio_value))

print(f'Log Portfolio Maximal Drawdown: {log_mdd:.2%}')
print(f'Buy and Hold portfolio Maximal Drawdown: {buy_and_hold_mdd:.2%}')
print(f'Market Portfolio Maximal Drawdown: {market_mdd:.2%}')

# Step 2: Calculation and Plotting of Log W_t for Different Portfolios
def calculate_w_t(log_V, T):
    W = np.zeros(len(log_V))
    V = np.exp(log_V)  # Convert log values back to original scale
    
    for t in range(len(log_V)):
        integral_sum = np.sum(V[:t+1])  # Sum of V_s from 0 to t (inclusive)
        W[t] = (1 / T) * integral_sum + (V[t] * (T - (t + 1)) / T)  # Formula for W_t
    return safe_log(W)

T = len(log_portfolio_value)  # Total time periods

# Calculate W_t for each portfolio
log_W_rebalanced = calculate_w_t(log_portfolio_value, T)
log_W_buy_and_hold = calculate_w_t(log_buy_and_hold_value, T)
log_W_market = calculate_w_t(log_market_portfolio_value, T)

# Plotting W_t for different portfolios
plt.figure(figsize=(14, 7))
plt.plot(log_W_rebalanced, label='Log W_t (Rebalanced)')
plt.plot(log_W_buy_and_hold, label='Log W_t (Buy and Hold)')
plt.plot(log_W_market, label='Log W_t (Market)')
plt.legend()
plt.title('Log W_t for Different Portfolios')
plt.xlabel('Time')
plt.ylabel('Log W_t')
plt.grid(True)
plt.show()

# Calculation of MDD for W_t
def calculate_mdd_for_w_t(log_W):
    series = pd.Series(np.exp(log_W))
    mdd = calculate_mdd(series)
    return mdd

mdd_W_rebalanced = calculate_mdd_for_w_t(log_W_rebalanced)
mdd_W_buy_and_hold = calculate_mdd_for_w_t(log_W_buy_and_hold)
mdd_W_market = calculate_mdd_for_w_t(log_W_market)

print(f'Log W_t (Rebalanced) Maximal Drawdown: {mdd_W_rebalanced:.2%}')
print(f'Log W_t (Buy and Hold) Maximal Drawdown: {mdd_W_buy_and_hold:.2%}')
print(f'Log W_t (Market) Maximal Drawdown: {mdd_W_market:.2%}')

# Step 3: Calculation and Plotting for CK Allocations with Different Alpha Values
def ck_allocation(returns, alpha, reg=1e-4):
    cum_returns = returns.cumsum()
    raw_covariance = np.dot(cum_returns.T, cum_returns)
    reg_covariance = raw_covariance + np.eye(raw_covariance.shape[0]) * reg
    inv_reg_covariance = np.linalg.inv(reg_covariance)
    raw_allocation = alpha * np.dot(inv_reg_covariance, cum_returns.iloc[-1].values)
    raw_allocation[raw_allocation < 0] = 0
    if raw_allocation.sum() > 1:
        allo = raw_allocation / raw_allocation.sum()
    else:
        allo = raw_allocation
    return allo

alphas = [1, 0.5, 0.1]
Z = len(returns)
V = {alpha: np.ones(Z) for alpha in alphas}
theta_ck = {alpha: np.zeros((Z, m)) for alpha in alphas}  # Initialize theta_ck

for t in range(1, Z):
    for a in alphas:
        allocation1 = ck_allocation(returns.iloc[:t], a)
        theta_ck[a][t] = allocation1  # Store the allocation weights
        portfolio_return = (allocation1 * returns.iloc[t]).sum()
        V[a][t] = V[a][t-1] * (1 + portfolio_return)

V_df = pd.DataFrame(V, index=returns.index)

# Plotting Log V_t for different alpha values
plt.figure(figsize=(14, 7))
for alpha in alphas:
    plt.plot(safe_log(V_df[alpha]), label=f'Log of Vt for α={alpha}')
plt.title('Log of Vt Values for Selected α-CK Allocations Over Time')
plt.xlabel('Date')
plt.ylabel('Log of Vt')
plt.legend()
plt.grid(True)
plt.show()

# Calculation of MDD for CK Allocation Portfolios
def calculate_mdd_for_ck(log_V_ck):
    return calculate_mdd(pd.Series(np.exp(log_V_ck)))

mdd_ck = {}
for alpha in alphas:
    mdd_ck[alpha] = calculate_mdd_for_ck(safe_log(V_df[alpha]))
    print(f'CK Allocation Portfolio (alpha={alpha}) Maximal Drawdown: {mdd_ck[alpha]:.2%}')

# Step 4: Produce the θ-Plot(s)
def plot_theta(theta_ck, assets):
    plt.figure(figsize=(14, 7))
    for i in range(len(assets)):
        plt.plot(theta_ck[:, i], label=f'Asset {assets[i]}')
    plt.plot(1 - theta_ck.sum(axis=1), label='Cash', linestyle='--')
    plt.legend()
    plt.title(f'CK Allocation Weights over Time')
    plt.xlabel('Time')
    plt.ylabel('Weights')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

for alpha in alphas:
    plot_theta(theta_ck[alpha], assets)

# Step 5: Define the function for calculating log W_t for CK Allocation Portfolios
def plot_log_w_t_for_ck(alpha_values, V_df):
    plt.figure(figsize=(14, 7))
    for alpha in alpha_values:
        log_W = calculate_w_t(safe_log(V_df[alpha]), len(V_df))
        plt.plot(log_W, label=f'Log W_t (α={alpha})')
    
    plt.title('Log W_t for CK Allocations')
    plt.xlabel('Time')
    plt.ylabel('Log W_t')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot log W_t for CK Allocation Portfolios
plot_log_w_t_for_ck(alphas, V_df)

# Calculate MDD for log W_t for CK Allocation Portfolios
def calculate_mdd_for_log_w_t_ck(alpha_values, V_df):
    mdd_w_t_ck = {}
    for alpha in alpha_values:
        log_W_ck = calculate_w_t(safe_log(V_df[alpha]), len(V_df))
        mdd_w_t_ck[alpha] = calculate_mdd(pd.Series(np.exp(log_W_ck)))
        print(f'CK Allocation Portfolio log W_t (alpha={alpha}) Maximal Drawdown: {mdd_w_t_ck[alpha]:.2%}')

# Call the function with the alpha values and V_df
calculate_mdd_for_log_w_t_ck(alphas, V_df)

# Step 6: Cover's Universal Portfolio
n_assets = returns.shape[1]
V_cover = np.ones(len(returns))

for t in range(1, len(returns)):
    avg_return = returns.iloc[:t].mean()
    optimal_weights = np.maximum(avg_return, 0) / np.sum(np.maximum(avg_return, 0))
    if np.sum(optimal_weights) == 0:
        optimal_weights = np.ones(n_assets) / n_assets
    portfolio_return = (optimal_weights * returns.iloc[t]).sum()
    V_cover[t] = V_cover[t-1] * (1 + portfolio_return)

# Handling zero values in V_cover before taking the logarithm
V_cover_nonzero = np.where(V_cover > 0, V_cover, np.nan)  # Replace zeros with NaNs to avoid log(0)

# Plotting the Log of Cover's Portfolio Value
plt.figure(figsize=(14, 7))
plt.plot(safe_log(V_cover_nonzero), label="Log of Vt for Cover's Portfolio")
plt.title("Log of Vt for Cover's Portfolio Over Time")
plt.xlabel("Date")
plt.ylabel("Log of Vt")
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print MDD for Cover's Portfolio (ignoring zeros)
cover_mdd = calculate_mdd(pd.Series(V_cover[V_cover > 0]))
print(f"Cover's Portfolio Maximal Drawdown: {cover_mdd:.2%}")

# Step 7: Defining V1 and V2

# V1 is a pure cash portfolio, which means it stays constant at 1
V1 = np.ones(len(returns))

# V2 is the CK Allocation with alpha = 10
alpha_10 = 10
V2 = np.ones(len(returns))  # Initialize V2 with 1s

for t in range(1, len(returns)):
    allocation_ck = ck_allocation(returns.iloc[:t], alpha_10)
    portfolio_return = (allocation_ck * returns.iloc[t]).sum()
    V2[t] = V2[t-1] * (1 + portfolio_return)

# Now plot V1 and V2
plt.figure(figsize=(14, 7))

# Log transformation for V1 and V2
log_V1 = np.log(V1)  # Pure Cash, stays at log(1) = 0
log_V2 = np.log(V2)  # CK Allocation with α=10

# Plot V1 and V2
plt.plot(log_V1, label="V1 (Pure Cash, log(1) = 0)")
plt.plot(log_V2, label="V2 (CK Allocation α=10)")

# Add titles and labels
plt.title("Log-Scaled Plot of Aggregate Portfolios V1 and V2")
plt.xlabel("Time")
plt.ylabel("Log Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

alpha = 10
V2 = np.ones(len(returns))
for t in range(1, len(returns)):
    allocation = ck_allocation(returns.iloc[:t], alpha)
    portfolio_return = (allocation * returns.iloc[t]).sum()
    V2[t] = V2[t-1] * (1 + portfolio_return)
log_ck_allocation_value = np.log(V2)

# Step 8: Modify Aggregating Portfolio
# Using Market portfolio (V1) and CK allocation (V2)
min_length = min(len(np.exp(log_market_portfolio_value)), len(V2))
V1_aligned = np.exp(log_market_portfolio_value)[:min_length]
V2_aligned = V2[:min_length]

def aggregate_portfolios(V1, V2):
    Z = len(V1)
    V_aggregated = np.zeros(Z)
    for t in range(Z):
        V_aggregated[t] = 0.5 * V1[t] + 0.5 * V2[t]  # Averaging V1 and V2
    return V_aggregated

V_aggregated = aggregate_portfolios(V1_aligned, V2_aligned)
log_V_aggregated = np.log(V_aggregated)

# MDD for each portfolio
mdd_market = calculate_mdd(np.exp(log_market_portfolio_value[:min_length]))
mdd_buy_and_hold = calculate_mdd(np.exp(log_buy_and_hold_value[:min_length]))
mdd_rebalancing = calculate_mdd(np.exp(log_portfolio_value[:min_length]))
mdd_ck_allocation = calculate_mdd(np.exp(log_ck_allocation_value[:min_length]))
mdd_aggregated = calculate_mdd(np.exp(log_V_aggregated))

# Plotting Comparison (Time Range from 0 to 500)
time_range = np.arange(min_length)

plt.figure(figsize=(14, 7))
plt.plot(time_range, log_market_portfolio_value[:min_length], label=f'Market Portfolio')
plt.plot(time_range, log_buy_and_hold_value[:min_length], label=f'Buy and Hold Portfolio')
plt.plot(time_range, log_portfolio_value[:min_length], label=f'Rebalancing Portfolio')
plt.plot(time_range, log_ck_allocation_value[:min_length], label=f'CK Allocation (α=10)')
plt.plot(time_range, log_V_aggregated, label=f'Aggregated Portfolio')

plt.legend()
plt.title('Comparison of Portfolios')
plt.xlabel('Time (weeks)')
plt.ylabel('Log Portfolio Value')
plt.grid(True)
plt.show()

# Display MDD after the plot
print(f'Market Portfolio MDD: {mdd_market:.2%}')
print(f'Buy and Hold Portfolio MDD: {mdd_buy_and_hold:.2%}')
print(f'Rebalancing Portfolio MDD: {mdd_rebalancing:.2%}')
print(f'CK Allocation Portfolio (α=10) MDD: {mdd_ck_allocation:.2%}')
print(f'Aggregated Portfolio MDD: {mdd_aggregated:.2%}')

# Step 9: Modify Aggregating Portfolio
# Using Buy and hold (V1) and CK allocation (V2)
min_length = min(len(np.exp(log_buy_and_hold_value)), len(V2))
V1_aligned = np.exp(log_buy_and_hold_value)[:min_length]
V2_aligned = V2[:min_length]

def aggregate_portfolios(V1, V2):
    Z = len(V1)
    V_aggregated = np.zeros(Z)
    for t in range(Z):
        V_aggregated[t] = 0.5 * V1[t] + 0.5 * V2[t]  # Averaging V1 and V2
    return V_aggregated

V_aggregated = aggregate_portfolios(V1_aligned, V2_aligned)
log_V_aggregated = np.log(V_aggregated)

# MDD for each portfolio
mdd_market = calculate_mdd(np.exp(log_market_portfolio_value[:min_length]))
mdd_buy_and_hold = calculate_mdd(np.exp(log_buy_and_hold_value[:min_length]))
mdd_rebalancing = calculate_mdd(np.exp(log_portfolio_value[:min_length]))
mdd_ck_allocation = calculate_mdd(np.exp(log_ck_allocation_value[:min_length]))
mdd_aggregated = calculate_mdd(np.exp(log_V_aggregated))

# Plotting Comparison (Time Range from 0 to 500)
time_range = np.arange(min_length)

plt.figure(figsize=(14, 7))
plt.plot(time_range, log_market_portfolio_value[:min_length], label=f'Market Portfolio')
plt.plot(time_range, log_buy_and_hold_value[:min_length], label=f'Buy and Hold Portfolio')
plt.plot(time_range, log_portfolio_value[:min_length], label=f'Rebalancing Portfolio')
plt.plot(time_range, log_ck_allocation_value[:min_length], label=f'CK Allocation (α=10)')
plt.plot(time_range, log_V_aggregated, label=f'Aggregated Portfolio')

plt.legend()
plt.title('Comparison of Portfolios')
plt.xlabel('Time (weeks)')
plt.ylabel('Log Portfolio Value')
plt.grid(True)
plt.show()

# Display MDD after the plot
print(f'Market Portfolio MDD: {mdd_market:.2%}')
print(f'Buy and Hold Portfolio MDD: {mdd_buy_and_hold:.2%}')
print(f'Rebalancing Portfolio MDD: {mdd_rebalancing:.2%}')
print(f'CK Allocation Portfolio (α=10) MDD: {mdd_ck_allocation:.2%}')
print(f'Aggregated Portfolio MDD: {mdd_aggregated:.2%}')

# Step 10: Modify Aggregating Portfolio
# Using Rebalancing (V1) and CK allocation (V2)
min_length = min(len(np.exp(log_portfolio_value)), len(V2))
V1_aligned = np.exp(log_portfolio_value)[:min_length]
V2_aligned = V2[:min_length]

def aggregate_portfolios(V1, V2):
    Z = len(V1)
    V_aggregated = np.zeros(Z)
    for t in range(Z):
        V_aggregated[t] = 0.5 * V1[t] + 0.5 * V2[t]  # Averaging V1 and V2
    return V_aggregated

V_aggregated = aggregate_portfolios(V1_aligned, V2_aligned)
log_V_aggregated = np.log(V_aggregated)

# MDD for each portfolio
mdd_market = calculate_mdd(np.exp(log_market_portfolio_value[:min_length]))
mdd_buy_and_hold = calculate_mdd(np.exp(log_buy_and_hold_value[:min_length]))
mdd_rebalancing = calculate_mdd(np.exp(log_portfolio_value[:min_length]))
mdd_ck_allocation = calculate_mdd(np.exp(log_ck_allocation_value[:min_length]))
mdd_aggregated = calculate_mdd(np.exp(log_V_aggregated))

# Plotting Comparison (Time Range from 0 to 500)
time_range = np.arange(min_length)

plt.figure(figsize=(14, 7))
plt.plot(time_range, log_market_portfolio_value[:min_length], label=f'Market Portfolio')
plt.plot(time_range, log_buy_and_hold_value[:min_length], label=f'Buy and Hold Portfolio')
plt.plot(time_range, log_portfolio_value[:min_length], label=f'Rebalancing Portfolio')
plt.plot(time_range, log_ck_allocation_value[:min_length], label=f'CK Allocation (α=10)')
plt.plot(time_range, log_V_aggregated, label=f'Aggregated Portfolio')

plt.legend()
plt.title('Comparison of Portfolios')
plt.xlabel('Time (weeks)')
plt.ylabel('Log Portfolio Value')
plt.grid(True)
plt.show()

# Display MDD after the plot
print(f'Market Portfolio MDD: {mdd_market:.2%}')
print(f'Buy and Hold Portfolio MDD: {mdd_buy_and_hold:.2%}')
print(f'Rebalancing Portfolio MDD: {mdd_rebalancing:.2%}')
print(f'CK Allocation Portfolio (α=10) MDD: {mdd_ck_allocation:.2%}')
print(f'Aggregated Portfolio MDD: {mdd_aggregated:.2%}')

# Rebalancing Portfolio (V1)
log_rebalancing_portfolio_value = safe_log(1 + returns.mul(theta).sum(axis=1)).cumsum()


# Step 11: Aggregating Portfolio (Using V1 = 1 and CK Allocation as V2)

def aggregate_portfolios(V1, V2):
    Z = len(V1)
    V_aggregated = np.zeros(Z)
    for t in range(Z):
        V_aggregated[t] = 0.5 * V1[t] + 0.5 * V2[t]  # Averaging V1 and V2
    return V_aggregated

V_aggregated = aggregate_portfolios(V1, V2)
log_V_aggregated = np.log(V_aggregated)

# Ensure all portfolios have the same length by trimming to the minimum length
min_length = min(len(log_market_portfolio_value), len(log_V_aggregated), 
                 len(log_rebalancing_portfolio_value), len(log_buy_and_hold_value), len(log_ck_allocation_value))

log_market_portfolio_value = log_market_portfolio_value[:min_length]
log_V_aggregated = log_V_aggregated[:min_length]
log_rebalancing_portfolio_value = log_rebalancing_portfolio_value[:min_length]
log_buy_and_hold_value = log_buy_and_hold_value[:min_length]
log_ck_allocation_value = log_ck_allocation_value[:min_length]
time_range = np.arange(min_length)

# Plotting Comparison of Portfolios

fig, ax = plt.subplots(figsize=(14, 7))

# Plot Pure Cash (V1)
ax.plot(time_range, np.log(V1[:min_length]), label=f'Pure Cash (V1)')

# Plot CK Allocation Portfolio (V2)
ax.plot(time_range, log_ck_allocation_value, label=f'CK Allocation (α=10)')

# Plot Aggregated Portfolio
ax.plot(time_range, log_V_aggregated, label=f'Aggregated Portfolio (Cash + CK)')

# Plot Market Portfolio
ax.plot(time_range, log_market_portfolio_value, label=f'Market Portfolio')

# Plot Buy and Hold Portfolio
ax.plot(time_range, log_buy_and_hold_value, label=f'Buy and Hold Portfolio')

# Plot Rebalancing Portfolio
ax.plot(time_range, log_rebalancing_portfolio_value, label=f'Rebalancing Portfolio')

plt.legend()
plt.title('Comparison of Portfolios')
plt.xlabel('Time (weeks)')
plt.ylabel('Log Portfolio Value')
plt.grid(True)
plt.show()

# Maximal Drawdown (MDD) Calculation

def calculate_mdd(series):
    # Convert NumPy array to pandas Series for cummax()
    series = pd.Series(series)
    peak = series.cummax()
    drawdown = (series - peak) / peak
    return drawdown.min()

# Calculate MDD for each portfolio
mdd_cash = calculate_mdd(np.exp(np.log(V1[:min_length])))  # MDD for pure cash is trivially 0
mdd_ck_allocation = calculate_mdd(np.exp(log_ck_allocation_value))
mdd_aggregated = calculate_mdd(np.exp(log_V_aggregated))
mdd_market = calculate_mdd(np.exp(log_market_portfolio_value))
mdd_buy_and_hold = calculate_mdd(np.exp(log_buy_and_hold_value))
mdd_rebalancing = calculate_mdd(np.exp(log_rebalancing_portfolio_value))

# Display MDD After the Plot
print(f'Pure Cash MDD: {mdd_cash:.2%}')
print(f'CK Allocation Portfolio MDD: {mdd_ck_allocation:.2%}')
print(f'Aggregated Portfolio MDD: {mdd_aggregated:.2%}')
print(f'Market Portfolio MDD: {mdd_market:.2%}')
print(f'Buy and Hold Portfolio MDD: {mdd_buy_and_hold:.2%}')
print(f'Rebalancing Portfolio MDD: {mdd_rebalancing:.2%}')

# Step 12: Learning alpha part 2: Produce V(α) and θ(α) for every α ∈ [0, 10]
def learning_alpha(returns, alpha_values):
    Z = len(returns)
    V_alpha = {alpha: np.ones(Z) for alpha in alpha_values}
    theta_alpha = {alpha: np.zeros((Z, len(returns.columns))) for alpha in alpha_values}
    
    for t in range(1, Z):
        for alpha in alpha_values:
            allocation = ck_allocation(returns.iloc[:t], alpha)
            theta_alpha[alpha][t] = allocation
            portfolio_return = (allocation * returns.iloc[t]).sum()
            V_alpha[alpha][t] = V_alpha[alpha][t-1] * (1 + portfolio_return)
    
    return V_alpha, theta_alpha

alpha_range = np.linspace(0, 10, 21)
V_alpha, theta_alpha = learning_alpha(returns, alpha_range)

# Plotting V(α) for different α values
plt.figure(figsize=(14, 7))
for alpha in alpha_range:
    plt.plot(V_alpha[alpha], label=f"α={alpha:.1f}")
plt.title("V(α) for Different α Values")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

# Define the function to compute CK allocation with regularization to avoid singular matrix
def ck_allocation(returns, alpha, reg=1e-4):
    cum_returns = returns.cumsum()  # Cumulative sum of returns
    raw_covariance = np.dot(cum_returns.T, cum_returns)  # Covariance matrix
    reg_covariance = raw_covariance + np.eye(raw_covariance.shape[0]) * reg  # Regularized covariance
    inv_reg_covariance = np.linalg.inv(reg_covariance)  # Inverse of regularized covariance matrix
    raw_allocation = alpha * np.dot(inv_reg_covariance, cum_returns.iloc[-1].values)  # CK allocation formula
    raw_allocation[raw_allocation < 0] = 0  # No negative allocations (long-only constraint)
    if raw_allocation.sum() > 1:  # Normalize allocation if the sum exceeds 1
        allo = raw_allocation / raw_allocation.sum()
    else:
        allo = raw_allocation
    return allo

# Step 13: Function to compute V(alpha) and θ(alpha) for each alpha in the given range [0, 10]
def learning_alpha(returns, alpha_values):
    Z = len(returns)  # Number of time periods
    m = returns.shape[1]  # Number of assets

    V_alpha = {alpha: np.ones(Z) for alpha in alpha_values}  # Initialize portfolio values for each alpha
    theta_alpha = {alpha: np.zeros((Z, m)) for alpha in alpha_values}  # Initialize allocation weights for each alpha

    for t in range(1, Z):  # For each time period
        for alpha in alpha_values:  # For each alpha
            allocation = ck_allocation(returns.iloc[:t], alpha)  # Calculate CK allocation for time t
            theta_alpha[alpha][t] = allocation  # Store the allocation weights
            portfolio_return = (allocation * returns.iloc[t]).sum()  # Calculate portfolio return at time t
            V_alpha[alpha][t] = V_alpha[alpha][t-1] * (1 + portfolio_return)  # Update portfolio value V(alpha)
    
    return V_alpha, theta_alpha

# Define the range of alpha values (e.g., 0 to 10 with 21 steps)
alpha_range = np.linspace(0, 10, 21)

# Call the function to compute V(alpha) and θ(alpha) for each alpha in the range
V_alpha, theta_alpha = learning_alpha(returns, alpha_range)

# Plot the allocation weights (θ(α)) over time for a few alpha values
def plot_theta(alpha, theta_alpha, assets):
    plt.figure(figsize=(14, 7))
    for i in range(len(assets)):  # Plot the weights for each asset
        plt.plot(theta_alpha[alpha][:, i], label=f'Asset {assets[i]}')
    plt.plot(1 - theta_alpha[alpha].sum(axis=1), label='Cash', linestyle='--')  # Plot the cash position
    plt.legend()
    plt.title(f'CK Allocation Weights (θ) for α={alpha:.1f}')
    plt.xlabel('Time')
    plt.ylabel('Weights')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
    
    # Choose a few alpha values to plot θ(α) over time
for alpha in [1, 5, 10]:
    plot_theta(alpha, theta_alpha, assets)
    
    # Step 14: Compute Cover's Portfolio over alpha in U = [0, 10]

def compute_covers_portfolio_over_alpha(V_alpha, alpha_values):
    Z = len(next(iter(V_alpha.values())))  # Total number of time periods
    cover_portfolio = np.zeros(Z)

    for t in range(Z):
        # Averaging V(alpha) over the range of alpha (approximation of the integral in equation (8))
        cover_portfolio[t] = np.mean([V_alpha[alpha][t] for alpha in alpha_values])

    return cover_portfolio

# Compute Cover's portfolio over alpha in U = [0, 10]
alpha_range = np.linspace(0, 10, 21)  # Create the range for alpha
V_cover_over_alpha = compute_covers_portfolio_over_alpha(V_alpha, alpha_range)

# Plotting the log of Cover's portfolio over alpha
plt.figure(figsize=(14, 7))
plt.plot(np.log(V_cover_over_alpha), label="Log of Cover's Portfolio (Alpha in U=[0, 10])", color="blue")
plt.plot(log_market_portfolio_value, label="Log of Market Portfolio V(t)", color="red", linestyle="--")
plt.title("Cover's Portfolio Over Alpha in U=[0, 10] vs Market Portfolio")
plt.xlabel("Time")
plt.ylabel("Log Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print MDD for Cover's Portfolio over alpha in U = [0, 10]
cover_mdd_over_alpha = calculate_mdd(np.exp(np.log(V_cover_over_alpha)))
print(f"Cover's Portfolio (Alpha in U=[0, 10]) Maximal Drawdown: {cover_mdd_over_alpha:.2%}")

    
    # Step 15: Compute Cover's Portfolio Value (Equation 8)
def compute_covers_portfolio(V_alpha, alpha_values):
    Z = len(next(iter(V_alpha.values())))  # Total number of time periods
    cover_portfolio = np.zeros(Z)
    
    for t in range(Z):
        # Averaging V(alpha) over the range of alpha (approximation of the integral in equation (8))
        cover_portfolio[t] = np.mean([V_alpha[alpha][t] for alpha in alpha_values])
    
    return cover_portfolio

# Compute Cover's Portfolio Allocation (Equation 9)
def compute_covers_allocation(V_alpha, theta_alpha, alpha_values):
    Z = len(next(iter(V_alpha.values())))  # Total number of time periods
    m = theta_alpha[alpha_values[0]].shape[1]  # Number of assets
    cover_allocation = np.zeros((Z, m))  # To store allocation weights over time
    
    for t in range(1, Z):
        numerator = np.zeros(m)  # For weighted sum of allocations
        denominator = 0  # For sum of V(alpha)
        
        for alpha in alpha_values:
            V_t_minus_1 = V_alpha[alpha][t-1]  # V(alpha) at time t-1
            numerator += theta_alpha[alpha][t] * V_t_minus_1  # Weight each allocation by V(alpha)
            denominator += V_t_minus_1  # Sum of V(alpha) for normalization
        
        cover_allocation[t] = numerator / denominator if denominator != 0 else np.zeros(m)  # Avoid division by zero
    
    return cover_allocation

# Step 16: Compute Cover's Portfolio and Compare with Market Portfolio
# Compute Cover's Portfolio Value
V_cover = compute_covers_portfolio(V_alpha, alpha_range)

# Compute Cover's Portfolio Allocation
theta_cover = compute_covers_allocation(V_alpha, theta_alpha, alpha_range)

# Plot Cover's Portfolio Value alongside Market and Individual Portfolios
plt.figure(figsize=(14, 7))
plt.plot(np.log(V_cover), label="Log of Cover's Portfolio V(t)", color="blue")
plt.plot(log_market_portfolio_value, label="Log of Market Portfolio V(t)", color="red", linestyle="--")

# Plot some individual V(α) portfolios for comparison
valid_alphas = [0, 1, 5, 10]  # Adjust alpha values to match those in your alpha_range
for alpha in valid_alphas:
    plt.plot(np.log(V_alpha[alpha]), label=f'Log of V(α={alpha})', linestyle=":")

plt.title("Comparison of Cover's Portfolio, Market Portfolio, and Individual V(α)")
plt.xlabel("Time")
plt.ylabel("Log Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

# Compute and print the MDD for Cover's portfolio (Simplex)
V_cover_mdd = calculate_mdd(np.exp(np.log(V_cover)))  # Using exponential to revert log scale
print(f"Cover's Portfolio Maximal Drawdown: {V_cover_mdd:.2%}")

# Step 17: Compare Allocations for Cover's Portfolio
def plot_theta_cover(theta_cover, assets):
    plt.figure(figsize=(14, 7))
    for i in range(len(assets)):
        plt.plot(theta_cover[:, i], label=f'Asset {assets[i]}')
    plt.plot(1 - theta_cover.sum(axis=1), label='Cash', linestyle='--')
    plt.legend()
    plt.title("Cover's Portfolio Allocation (θ) Over Time")
    plt.xlabel("Time")
    plt.ylabel("Allocation Weights")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

# Plot the allocation for the Cover's portfolio
plot_theta_cover(theta_cover, assets)

# Step 18: Generate portfolios in the m-dimensional simplex
def generate_simplex_weights(m, n_samples):
    """Generates n_samples points in the m-dimensional simplex."""
    weights = np.random.dirichlet(np.ones(m), size=n_samples)
    return weights

# Compute the constant rebalanced portfolio value for each b in the simplex
def constant_rebalanced_portfolio(returns, b):
    """Computes the portfolio value using constant weights b over time."""
    Z = len(returns)
    V = np.ones(Z)  # Initialize portfolio value with 1
    for t in range(1, Z):
        portfolio_return = (b * returns.iloc[t]).sum()  # Compute portfolio return at time t
        V[t] = V[t-1] * (1 + portfolio_return)  # Update portfolio value
    return V

# Use Cover's algorithm to produce the portfolio where U is the simplex
def covers_portfolio_simplex(returns, m, n_samples=1000):
    """Computes Cover's portfolio by averaging over all portfolios in the simplex."""
    weights = generate_simplex_weights(m, n_samples)  # Generate weights in the simplex
    Z = len(returns)
    V_cover = np.zeros(Z)  # Initialize Cover's portfolio value

    for b in weights:
        V_b = constant_rebalanced_portfolio(returns, b)  # Portfolio value for weight vector b
        V_cover += V_b  # Accumulate over all portfolios
    
    V_cover /= n_samples  # Average to form Cover's portfolio
    return V_cover

# Compute and compare the Cover's portfolio with market and other portfolios
# Assuming `returns` is the DataFrame of asset returns
m = returns.shape[1]  # Number of assets

# Compute Cover's portfolio using the simplex
V_cover_simplex = covers_portfolio_simplex(returns, m)

# Plot Cover's portfolio value alongside Market and Individual Portfolios
plt.figure(figsize=(14, 7))
plt.plot(np.log(V_cover_simplex), label="Log of Cover's Portfolio (Simplex)", color="blue")
plt.plot(log_market_portfolio_value, label="Log of Market Portfolio V(t)", color="red", linestyle="--")

# Plot some individual V(α) portfolios for comparison
valid_alphas = [0, 1, 5, 10]  # Adjust alpha values to match those in your alpha_range
for alpha in valid_alphas:
    plt.plot(np.log(V_alpha[alpha]), label=f'Log of V(α={alpha})', linestyle=":")

plt.title("Comparison of Cover's Portfolio (Simplex), Market Portfolio, and Individual V(α)")
plt.xlabel("Time")
plt.ylabel("Log Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

# Compute and print the MDD for Cover's portfolio (Simplex)
cover_simplex_mdd = calculate_mdd(np.exp(np.log(V_cover_simplex)))  # Using exponential to revert log scale
print(f"Cover's Portfolio (Simplex) Maximal Drawdown: {cover_simplex_mdd:.2%}")


# In[ ]:




