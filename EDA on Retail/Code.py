import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df_retail = pd.read_csv(r'C:\Users\priya\Documents\DA - Infobyte\EDA on Retail\Online_Retail.csv', encoding='ISO-8859-1')

# overview of dataset
print("The shape of our dataset is: ", df_retail.shape)
df_retail.info()

print(df_retail.head())
print(df_retail.tail())

# Exploring unique values of each attribute
# print("Number of transactions: ", df_retail['InvoiceNo'].nunique())
# print("Number of products: ", df_retail['StockCode'].nunique())
# print("Number of customers:", df_retail['CustomerID'].nunique())
# print("Percentage of customers NA: ", round(df_retail['CustomerID'].isnull().sum() * 100 / len(df_retail), 2), "%")
print(df_retail.nunique())
# Conversion of column types
df_retail[['Quantity', 'UnitPrice']] = df_retail[['Quantity', 'UnitPrice']].astype('float')
df_retail['TotalAmount'] = df_retail['Quantity'] * df_retail['UnitPrice']

df_retail['InvoiceDate'] = pd.to_datetime(df_retail['InvoiceDate'], errors='coerce')

print("Number of invalid dates: ", df_retail['InvoiceDate'].isna().sum())
df_retail = df_retail.dropna(subset=['InvoiceDate'])
print(df_retail['InvoiceDate'].head())

print(df_retail.isnull().sum(axis=0).to_frame())
df_retail = df_retail.dropna()

print(df_retail.describe())
df_retail.info()

# Filter out canceled invoices
df_sales = df_retail[~df_retail['InvoiceNo'].str.startswith('C')]
df_sales = df_sales.reset_index(drop=True)
print("Number of rows after excluding canceled invoices: ", df_sales.shape[0])

# Converting numerical columns
df_sales[['Quantity', 'UnitPrice']] = df_sales[['Quantity', 'UnitPrice']].astype('float')
df_sales['TotalAmount'] = df_sales['Quantity'] * df_sales['UnitPrice']

# Converting date-time columns with explicit format
df_sales['InvoiceDate'] = pd.to_datetime(df_sales['InvoiceDate'], format='%d-%m-%Y', errors='coerce')
print(df_sales['InvoiceDate'].head())

print(df_sales.describe())

# Descriptive Statistics
columns_to_analyze = ['TotalAmount', 'Quantity', 'UnitPrice']

for column in columns_to_analyze:
    print(f"\nStatistics for '{column}':")
    statistics = {}
    
    statistics['Mean'] = df_sales[column].mean()
    statistics['Median'] = df_sales[column].median()
    statistics['Mode'] = df_sales[column].mode().values
    statistics['Standard Deviation'] = df_sales[column].std()
    
    for stat_name, value in statistics.items():
        print(f"{stat_name}: {value}")

# Check for duplicates
duplicates = df_sales.duplicated().sum()
df_sales = df_sales.drop_duplicates()

# Customer and Product analysis
# Orders per customer
orders_per_customer = df_sales.groupby('CustomerID')['InvoiceNo'].nunique()
print(orders_per_customer.head())
top_customers = df_sales.groupby('CustomerID')['TotalAmount'].sum().nlargest(10)
top_products = df_sales.groupby('StockCode')['Quantity'].sum().nlargest(10)
print("Top Customers by Revenue:\n", top_customers)
print("Top Products by Quantity Sold:\n", top_products)
print(df_sales.columns)

customer_country = df_retail.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False)
print(customer_country)

customer_spending = df_retail.groupby('CustomerID')['TotalAmount'].sum().sort_values(ascending=False)
print(customer_spending.head(10))

purchase_frequency = df_retail.groupby('CustomerID')['InvoiceNo'].count().sort_values(ascending=False)
print(purchase_frequency.head(10))

top_products = df_retail.groupby('StockCode')['TotalAmount'].sum().sort_values(ascending=False)
print(top_products.head(10))

plt.figure(figsize=(10, 6))
customer_country.head(10).plot(kind='bar')
plt.title('Top 10 Countries by Number of Customers')
plt.ylabel('Number of Customers')
plt.show()

plt.figure(figsize=(10, 6))
top_products.head(10).plot(kind='bar')
plt.title('Top 10 Products by Sales')
plt.ylabel('Total Sales')
plt.show()

df_retail['MonthYear'] = df_retail['InvoiceDate'].dt.to_period('M')
monthly_sales = df_retail.groupby('MonthYear')['TotalAmount'].sum()
monthly_sales.plot(kind='line', figsize=(10, 6))
plt.title('Monthly Sales Trend')
plt.ylabel('Total Sales')
plt.show()

# Time Series analysis
print(df_sales['InvoiceDate'].dtype)

# Resampling the data to get monthly total sales
df_sales.set_index('InvoiceDate', inplace=True)
monthly_sales = df_sales['TotalAmount'].resample('M').sum()

# Plotting the time series
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, marker='o', linestyle='-', label='Monthly Sales')
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.legend()
plt.show()

# Calculate and plot a 6-month moving average
moving_avg = monthly_sales.rolling(window=6).mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, marker='o', linestyle='-', label='Monthly Sales')
plt.plot(moving_avg, color='red', label='6-Month Moving Average')
plt.title('Monthly Sales with Moving Average')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.legend()
plt.show()
