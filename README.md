<!DOCTYPE html>
<html lang="en">

<body>

<h1>Time Series Analysis and Prediction</h1>

<p>This project demonstrates <strong>time series analysis, forecasting, and deep learning prediction techniques</strong> using Python. It explores statistical and deep learning approaches to analyze temporal patterns and generate forecasts.</p>

<h2>Project Overview</h2>
<ul>
    <li>Performed exploratory time series analysis and visualization.</li>
    <li>Tested stationarity and decomposed trend, seasonality, and residuals.</li>
    <li>Forecasted using ARIMA/SARIMA and LSTM neural networks.</li>
</ul>

<h2>Dataset</h2>
<p><strong>AirPassengers Dataset:</strong> Monthly total airline passengers from 1949 to 1960.</p>

<table>
    <tr>
        <th>Column</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>Date</td>
        <td>Month and Year</td>
    </tr>
    <tr>
        <td>Passengers</td>
        <td>Number of airline passengers</td>
    </tr>
</table>

<p><strong>Size:</strong> 144 observations (12 years of monthly data)</p>

<h2>Technologies Used</h2>
<ul>
    <li>Python</li>
    <li>NumPy, Pandas</li>
    <li>Matplotlib, Seaborn</li>
    <li>Scikit-learn</li>
    <li>Statsmodels, pmdarima</li>
    <li>TensorFlow / Keras</li>
</ul>

<h2>Project Workflow</h2>

<h3>I.Data Preprocessing</h3>
<p>Steps performed:</p>
<ul>
    <li>Load dataset</li>
    <li>Rename columns</li>
    <li>Convert date column to datetime format</li>
    <li>Visualize passenger trends</li>
</ul>

<pre><code>airpass = pd.read_csv("AirPassengers.csv")
airpass.rename(columns={'Month':'Date','#Passengers':'Passengers'}, inplace=True)
</code></pre>

<h3>II.Exploratory Data Analysis (EDA)</h3>
<p>Visualizing trends and seasonality:</p>
<ul>
    <li>Overall growth trend</li>
    <li>Seasonal patterns (repeating yearly)</li>
    <li>Monthly passenger variations</li>
</ul>

<h3>III.Stationarity Testing</h3>
<p>Used the Augmented Dickey-Fuller (ADF) test to check stationarity:</p>

<pre><code>from statsmodels.tsa.stattools import adfuller
result = adfuller(airpass['Passengers'])
</code></pre>

<p><strong>Interpretation:</strong> p-value &lt; 0.05 → stationary</p>

<h3>IV.Trend, Seasonality, and Residual Decomposition</h3>

<pre><code>from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(airpass['Passengers'], model='multiplicative', period=12)
</code></pre>

<h3>V.Auto-Correlation Analysis</h3>

<p>Used ACF and PACF plots to identify lag dependencies for ARIMA/SARIMA models.</p>

<h3>VI.Forecasting with ARIMA / SARIMA</h3>
<p>Auto ARIMA selects the best parameters, followed by training a SARIMAX model:</p>

<pre><code>from pmdarima import auto_arima
model = auto_arima(airpass['Passengers'], seasonal=True, m=12)
</code></pre>

<h3>VII.Forecasting with LSTM (Deep Learning)</h3>
<p>Steps:</p>
<ul>
    <li>Normalize dataset using MinMaxScaler</li>
    <li>Create input sequences</li>
    <li>Train LSTM model</li>
    <li>Predict future passenger demand</li>
</ul>

<pre><code>model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
</code></pre>

<h2>Model Evaluation</h2>
<p>Evaluated using <strong>RMSE (Root Mean Squared Error)</strong> for training and test predictions.</p>

<h2>Forecast Output</h2>
<ul>
    <li>Historical passenger trends</li>
    <li>Predicted passenger demand</li>
    <li>Future forecasts (3 years ahead)</li>
</ul>

<h2>Author</h2>
<p><strong>Muhammed Raihan</strong><br>
Bachelor's in Computer Science<br>
Specialization: Artificial Intelligence & Big Data</p>

</body>
</html>
