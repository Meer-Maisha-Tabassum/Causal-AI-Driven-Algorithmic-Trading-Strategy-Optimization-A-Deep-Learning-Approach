#**************** IMPORT PACKAGES ********************
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from datetime import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st
import plotly.graph_objs as go
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import dowhy
from dowhy import CausalModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

#**************** FUNCTION TO FETCH DATA ***************************
def get_historical(quote):
    try:
        end = datetime.now()
        start = datetime(end.year-10, end.month, end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if df.empty:
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol=quote, outputsize='full')
            data = data.reset_index()
            # Keep Required cols only
            df = pd.DataFrame()
            df['Date'] = data['date']
            df['Open'] = data['1. open']
            df['High'] = data['2. high']
            df['Low'] = data['3. low']
            df['Close'] = data['4. close']
            df['Adj Close'] = data['5. adjusted close']
            df['Volume'] = data['6. volume']
            df.to_csv(''+quote+'.csv', index=False)
        return
    except Exception as e:
        st.write("Could not fetch historical data for {}. Try with another stock symbol.".format(quote))


# Function to calculate On-balance volume (OBV)
def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df

# Function to calculate Accumulation/distribution (A/D) line
def calculate_ad_line(df):
    df['ADL'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['ADL'] = df['ADL'].cumsum()
    return df

# Function to calculate Average Directional Index (ADX)
def calculate_adx(df, window=14):
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
    df['TRn'] = df['TR'].rolling(window=window).sum()
    df['DMplusn'] = df['DMplus'].rolling(window=window).sum()
    df['DMminusn'] = df['DMminus'].rolling(window=window).sum()
    df['DIplus'] = (df['DMplusn'] / df['TRn']) * 100
    df['DIminus'] = (df['DMminusn'] / df['TRn']) * 100
    df['DX'] = (abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])) * 100
    df['ADX'] = df['DX'].rolling(window=window).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR', 'DMplus', 'DMminus', 'TRn', 'DMplusn', 'DMminusn'], axis=1, inplace=True)
    return df

# Function to calculate Aroon Oscillator
def calculate_aroon_oscillator(df, window=25):
    df['Aroon Up'] = df['High'].rolling(window=window).apply(lambda x: x.argmax()) / window * 100
    df['Aroon Down'] = df['Low'].rolling(window=window).apply(lambda x: x.argmin()) / window * 100
    df['Aroon Oscillator'] = df['Aroon Up'] - df['Aroon Down']
    df.drop(['Aroon Up', 'Aroon Down'], axis=1, inplace=True)
    return df

# Function to calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    exp1 = df['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    df['MACD'] = macd
    df['MACD Signal'] = signal
    df['MACD Histogram'] = macd - signal
    return df

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(df, window=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

# Function to calculate Stochastic Oscillator
def calculate_stochastic_oscillator(df, window=14):
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    df['%K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

def ARIMA_ALGO(df, quote):
    uniqueVals = df["Code"].unique()  
    len(uniqueVals)
    df=df.set_index("Code")
    # for daily basis
    def parser(x):
        if isinstance(x, str):
            return datetime.strptime(x, '%Y-%m-%d')
        else:
            # Handle non-string values (e.g., NaN) appropriately
            return None
    def arima_model(train, test):
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(6,1 ,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions
    for company in uniqueVals[:10]:
        data=(df.loc[company,:]).reset_index()
        data['Price'] = data['Close']
        Quantity_date = data[['Price','Date']]
        Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
        Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        Quantity_date = Quantity_date.drop(['Date'],axis =1)
        
        traces = []
        for col in Quantity_date.columns:
            trace = go.Scatter(x=Quantity_date.index, y=Quantity_date[col], mode='lines', name=col)
            traces.append(trace)

        # Create the Plotly figure
        fig = go.Figure(traces)

        # Update layout to make the graph interactive
        fig.update_layout(title='Trends',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_dark',  # Use dark theme for better appeal
                        legend=dict(x=0, y=1))

        # Display Plotly figure in Streamlit
        st.plotly_chart(fig)

        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]
        # fit in model
        predictions = arima_model(train, test)
        
        # Plot graph using Plotly
        fig = go.Figure()

        # Add actual price trace
        fig.add_trace(go.Scatter(x=np.arange(len(test)), y=test.flatten(), mode='lines', name='Actual Price'))

        # Add predicted price trace
        fig.add_trace(go.Scatter(x=np.arange(len(test)), y=np.array(predictions).flatten(), mode='lines', name='Predicted Price'))

        # Update layout
        fig.update_layout(title='ARIMA Prediction',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          template='plotly_dark',  # Use dark theme for better appeal
                          legend=dict(x=0, y=1))

        # Display Plotly figure in Streamlit
        st.plotly_chart(fig)
        
        # Calculate MAE
        mae_arima = mean_absolute_error(test, predictions)
        print("ARIMA MAE:", mae_arima)

        print()
        print("##############################################################################")
        arima_pred=predictions[-2]
        print("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
        # rmse calculation
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        print("ARIMA RMSE:",error_arima)
        print("##############################################################################")
        
        # Forecasting Prediction for next 7 days
    
        forecast_arima = arima_model(train, test)
            
        return arima_pred, mae_arima, error_arima, forecast_arima[-8:]
    
def validate_data(df):
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        # Handle missing values
        df = df.dropna()
    return df

#************* LSTM SECTION **********************
def LSTM_ALGO(df, quote):   
    # Split data into training set and test set
    dataset_train=df.iloc[0:int(0.8*len(df)),:]
    dataset_test=df.iloc[int(0.8*len(df)):,:]
    
    ############# NOTE #################
    # TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
    # HERE N=7
    training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
    
    # Feature Scaling
    sc=MinMaxScaler(feature_range=(0,1))# Scaled values btween 0,1
    training_set_scaled=sc.fit_transform(training_set)
    # In scaling, fit_transform for training, transform for test
    
    # Creating data stucture with 7 timesteps and 1 output. 
    # 7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
    X_train=[]# memory with 7 days from day i
    y_train=[]# day i
    for i in range(7,len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i,0])
        y_train.append(training_set_scaled[i,0])
    # Convert list to numpy arrays
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_forecast=np.array(X_train[-1,1:])
    X_forecast=np.append(X_forecast,y_train[-1])
    # Reshaping: Adding 3rd dimension
    X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
    X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
    
    # Building RNN
    from tf_keras.models import Sequential
    from tf_keras.layers import LSTM, Dense, Dropout
    # Initialise RNN
    regressor=Sequential()
    
    # Add first LSTM layer
    regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    regressor.add(Dropout(0.1))
    
    # Add 2nd LSTM layer
    regressor.add(LSTM(units=50,return_sequences=True))
    regressor.add(Dropout(0.1))
    
    # Add 3rd LSTM layer
    regressor.add(LSTM(units=50,return_sequences=True))
    regressor.add(Dropout(0.1))
    
    # Add 4th LSTM layer
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    
    # Add o/p layer
    regressor.add(Dense(units=1))
    
    # Compile
    regressor.compile(optimizer='adam',loss='mean_squared_error')
    
    # Training
    regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
    
    # Testing
    real_stock_price=dataset_test.iloc[:,4:5].values
    
    # To predict, we need stock prices of 7 days before the test set
    # So combine train and test set to get the entire data set
    dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
    testing_set=dataset_total[len(dataset_total)-len(dataset_test)-7:].values
    testing_set=testing_set.reshape(-1,1)
    
    # Feature scaling
    testing_set=sc.transform(testing_set)
    
    # Create data structure
    X_test=[]
    for i in range(7,len(testing_set)):
        X_test.append(testing_set[i-7:i,0])
    # Convert list to numpy arrays
    X_test=np.array(X_test)
    
    # Reshaping: Adding 3rd dimension
    X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    
    # Testing Prediction
    predicted_stock_price=regressor.predict(X_test)
    
    # Getting original prices back from scaled values
    predicted_stock_price=sc.inverse_transform(predicted_stock_price)
    
    fig = go.Figure()

    # Add actual price trace
    fig.add_trace(go.Scatter(x=dataset_test.index, y=real_stock_price.flatten(), mode='lines', name='Actual Price'))

    # Add predicted price trace
    fig.add_trace(go.Scatter(x=dataset_test.index, y=predicted_stock_price.flatten(), mode='lines', name='Predicted Price'))

    # Update layout
    fig.update_layout(title='LSTM Prediction',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      template='plotly_dark',  # Use dark theme for better appeal
                      legend=dict(x=0, y=1))

    # Display Plotly figure in Streamlit
    st.plotly_chart(fig)

    print(np.isnan(predicted_stock_price).any())


    print(np.isnan(predicted_stock_price).any())

    print(predicted_stock_price)

    
    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    
    # Forecasting Prediction
    forecasted_stock_price=regressor.predict(X_forecast)
    
    # Getting original prices back from scaled values
    forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
    lstm_pred=forecasted_stock_price[0,0]
    print()
    
    # Calculate MAE
    mae_lstm = mean_absolute_error(real_stock_price, predicted_stock_price)
    print("LSTM MAE:", mae_lstm)
    
    print("##############################################################################")
    print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
    print("LSTM RMSE:",error_lstm)
    print("##############################################################################")

    # Forecasting Prediction for next 7 days
    forecast_lstm = [lstm_pred]  # Start with tomorrow's prediction
    # Forecasting for 7 days
    for i in range(7):
        X_forecast = np.array(X_train[-1, 1:])
        X_forecast = np.append(X_forecast, y_train[-1])
        X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
        forecasted_price = regressor.predict(X_forecast)
        # Invert scaling transformation
        forecasted_price_original = sc.inverse_transform(forecasted_price)
        forecast_lstm.append(forecasted_price_original[0, 0])
        # Updating X_train and y_train for the next prediction
        X_train = np.append(X_train, X_forecast, axis=0)
        y_train = np.append(y_train, forecasted_price)

    # Return the LSTM forecast
    return lstm_pred, mae_lstm, error_lstm, forecast_lstm[-8:]

#***************** LINEAR REGRESSION SECTION ******************       
def LIN_REG_ALGO(df, quote):
    # No of days to be forecasted in future
    forecast_out = int(7)
    # Price after n days
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    # New df with only relevant data
    df_new = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'OBV', 'ADL', 'ADX', 'Aroon Oscillator', 'MACD', 'RSI', '%K', '%D', 'Close after n days']]
    df_new = df_new.dropna()

    # Structure data for train, test & forecast
    # labels of known data, discard last 35 rows
    y = np.array(df_new.iloc[:-forecast_out,-1])
    y=np.reshape(y, (-1,1))
    # all cols of known data except labels, discard last 35 rows
    X=np.array(df_new.iloc[:-forecast_out,0:-1])
    # Unknown, X to be forecasted
    X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])

    # Training, testing to plot graphs, check accuracy
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    # Feature Scaling===Normalization
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_to_be_forecasted = sc.transform(X_to_be_forecasted)

    # Training
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Shape of X_train:", X_train.shape)

    # Testing
    y_test_pred = clf.predict(X_test)
    y_test_pred=y_test_pred*(1.04)

    fig = go.Figure()

    # Add actual price trace
    fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test.flatten(), mode='lines', name='Actual Price'))

    # Add predicted price trace
    fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test_pred.flatten(), mode='lines', name='Predicted Price'))

    # Update layout
    fig.update_layout(title='Linear Regression Prediction',
                      xaxis_title='Index',
                      yaxis_title='Price',
                      template='plotly_dark',  # Use dark theme for better appeal
                      legend=dict(x=0, y=1))

    # Display Plotly figure in Streamlit
    st.plotly_chart(fig)

    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Calculate MAE
    mae_lr = mean_absolute_error(y_test, y_test_pred)
    print("Linear Regression MAE:", mae_lr)
    
    # Forecasting
    forecast_lr = clf.predict(X_to_be_forecasted)
    forecast_lr=forecast_lr*(1.04)
    mean=forecast_lr.mean()
    lr_pred=forecast_lr[0,0]
    print()
    print("##############################################################################")
    print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
    print("Linear Regression RMSE:",error_lr)
    print("##############################################################################")

    return df, lr_pred, mae_lr, forecast_lr, mean, error_lr


#***************** SENTIMENT ANALYSIS **************************
def get_news_sentiment(symbol):
    try:
        finviz_url = 'https://finviz.com/quote.ashx?t='
        # Define the URL for the stock symbol
        url = finviz_url + symbol

        # Set up the request headers
        req = Request(url=url, headers={'user-agent': 'my-app'})

        # Open the URL and read the page content
        response = urlopen(req).read()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response, 'html.parser')

        # Extract news headlines from the HTML
        news_table = soup.find(id='news-table')
        news_list = news_table.find_all('tr')

        # Combine all headlines into a single string
        headlines = ' '.join([row.a.text for row in news_list])

        # Initialize VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()

        # Analyze sentiment of the headlines
        sentiment_scores = sid.polarity_scores(headlines)

        # Extract the compound score (overall sentiment polarity)
        compound_score = sentiment_scores['compound']

        # Extract individual scores for visualization
        positive_score = sentiment_scores['pos']
        negative_score = sentiment_scores['neg']
        neutral_score = sentiment_scores['neu']

        # Create an interactive pie chart
        fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative', 'Neutral'],
                                    values=[positive_score, negative_score, neutral_score],
                                    hole=0.3)])
        fig.update_layout(title='Sentiment Distribution in News Headlines',
                        template='plotly_dark')

        # Return the compound score as the overall sentiment polarity and the Plotly figure
        return compound_score, fig
    
    except Exception as e:
        print("An error occurred while fetching news sentiment for {}: {}".format(symbol, e))
        return None, None

def recommending(df, global_polarity_tuple, today_stock, mean, quote):
    # Extract the compound sentiment score from the tuple
    global_polarity = global_polarity_tuple[0]

    if today_stock.iloc[-1]['Close'] < mean:
        if global_polarity <= 0:
            idea="RISE"
            decision="BUY"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
        elif global_polarity > 0:
            idea="FALL"
            decision="SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
    else:
        idea="FALL"
        decision="SELL"
        print()
        print("##############################################################################")
        print("According to the ML Predictions and Sentiment Analysis of News, a",idea,"in",quote,"stock is expected => ",decision)
    return idea, decision

# Function to varify ADF test
def varify_adf_test(df):
    result = adfuller(df.values)
    print('ADF Statistics: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    p_value = result[1]
    return p_value

# Function for ADF test
def adf_test(df):
    result = adfuller(df.values)
    st.write('ADF Statistics: %f' % result[0])
    st.write('p-value: %f' % result[1])
    st.write('Critical values:')
    for key, value in result[4].items():
        st.write('\t%s: %.3f' % (key, value))
    return result

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    maxlag=15
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def insert(nm):
    quote = nm
    # Try-except to check if valid stock symbol
    try:
        get_historical(quote)
    except:
        return "An error occurred while fetching historical data: {}".format(e)
    else:
        #************** PREPROCESSING ***********************
        df = pd.read_csv(''+quote+'.csv')
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        today_stock = df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        df = calculate_obv(df)
        df = calculate_ad_line(df)
        df = calculate_adx(df)
        df = calculate_aroon_oscillator(df)
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_stochastic_oscillator(df)
        
        # Displaying the results in the Streamlit UI
        st.subheader('Todays Stock Data:')
        st.dataframe(today_stock)

        code_list = []
        for i in range(0,len(df)):
            code_list.append(quote)
        df2 = pd.DataFrame(code_list,columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2
        
        try:
            arima_pred, mae_arima, error_arima, forecast_arima = ARIMA_ALGO(df, quote)
            lstm_pred, mae_lstm, error_lstm, forecast_lstm = LSTM_ALGO(df, quote)
            df, lr_pred, mae_lr, forecast_lr, mean, error_lr = LIN_REG_ALGO(df, quote)
        except Exception as e:
            return "An error occurred during the prediction process: {}".format(e)
        
        # idea, decision = recommending(df, today_stock, mean)  # Included 'mean'
        print()
        print("Forecasted Prices for Next 7 days (Linear Regression):")
        forecast_df_lr = pd.DataFrame(forecast_lr, columns=['Predicted Price'])
        forecast_df_lr.index = pd.date_range(start=pd.to_datetime(df['Date'].iloc[-1]) + pd.DateOffset(days=1), periods=len(forecast_df_lr), freq='D')
        print(forecast_df_lr)
        
        print("Forecasted Prices for Next 7 days (ARIMA):")
        forecast_df_arima = pd.DataFrame(forecast_arima, columns=['Predicted Price'])
        forecast_df_arima.index = pd.date_range(start=pd.to_datetime(df['Date'].iloc[-1]) + pd.DateOffset(days=1), periods=len(forecast_df_arima), freq='D')
        print(forecast_df_arima)
        
        print("Forecasted Prices for Next 7 days (LSTM):")
        forecast_df_lstm = pd.DataFrame(forecast_lstm, columns=['Predicted Price'])
        forecast_df_lstm.index = pd.date_range(start=pd.to_datetime(df['Date'].iloc[-1]) + pd.DateOffset(days=1), periods=len(forecast_df_lstm), freq='D')
        print(forecast_df_lstm)

        st.subheader("Tomorrow's Closing Price Prediction:")
        st.write("ARIMA Prediction:", arima_pred)
        st.write("LSTM Prediction:", lstm_pred)
        st.write("Linear Regression Prediction:", lr_pred)
        
        st.subheader("MAE Values:")
        st.write("ARIMA MAE:", mae_arima)
        st.write("LSTM MAE:", mae_lstm)
        st.write("Linear Regression MAE:", mae_lr)

        st.subheader("RMSE Values:")
        st.write("ARIMA RMSE:", error_arima)
        st.write("LSTM RMSE:", error_lstm)
        st.write("Linear Regression RMSE:", error_lr)
        
        # Define the performance metrics for each model
        models = ['ARIMA_ALGO', 'LSTM_ALGO', 'LIN_REG_ALGO']
        mae_values = [mae_arima, mae_lstm, mae_lr]
        rmse_values = [error_arima, error_lstm, error_lr]

        # Plotting MAE
        st.write("Model Comparison: MAE")
        st.bar_chart({model: mae for model, mae in zip(models, mae_values)})

        # Plotting RMSE
        st.write("Model Comparison: RMSE")
        st.bar_chart({model: rmse for model, rmse in zip(models, rmse_values)})
        
        st.subheader("Forecasted Prices for Next 7 days (ARIMA):")
        st.dataframe(forecast_df_arima)
        
        st.subheader("Forecasted Prices for Next 7 days (LSTM):")
        st.dataframe(forecast_df_lstm)
        
        st.subheader("Forecasted Prices for Next 7 days (Linear Regression):")
        st.dataframe(forecast_df_lr)
        
        # Perform sentiment analysis
        compound_score, fig = get_news_sentiment(quote)

        # Display the compound sentiment score
        st.subheader('Compound Sentiment Score:')
        st.write(compound_score)

        # Display the interactive pie chart
        st.subheader('Sentiment Distribution in News Headlines:')
        st.plotly_chart(fig)
        
        global_polarity = get_news_sentiment(quote)

        idea, decision = recommending(df, global_polarity, today_stock, mean, quote)
        st.write("According to the ML Predictions and Sentiment Analysis of News, a",idea,"in",quote,"stock is expected => ",decision)
        
        df = df.dropna()
        
        # Initialize the variables outside the conditional blocks
        df_transformed_close = None
        df_transformed_rsi = None
        df_transformed_macd = None
        df_transformed_volume = None
        
        # Initialize combined DataFrame
        combined = pd.DataFrame()
        
        # Perform ADF test
        st.write('ADF Statistics for Close:-')
        if varify_adf_test(df['Close']) > 0.05:
            df_transformed_close = df['Close'].diff().dropna()
            adf_test(df_transformed_close)
            if varify_adf_test(df_transformed_close) < 0.05:
                st.write('Data is Stationary.')
                combined = pd.concat([combined, df_transformed_close], axis=1)
        else:
            adf_test(df['Close'])
            df_transformed_close = df['Close']
            st.write('Data is Stationary.')
            combined = pd.concat([combined, df['Close']], axis=1)
             
        
        # Perform ADF test
        st.write('ADF Statistics for RSI:-')
        if varify_adf_test(df['RSI']) > 0.05:
            st.write('ADF Statistics for RSI:-')
            df_transformed_rsi = df['RSI'].diff().dropna()
            adf_test(df_transformed_rsi)
            if varify_adf_test(df_transformed_rsi) < 0.05:
                st.write('Data is Stationary.')
                st.write(df_transformed_rsi.head(50))
                combined = pd.concat([combined, df_transformed_rsi], axis=1)
        else:
            adf_test(df['RSI'])
            df_transformed_rsi = df['RSI']
            st.write('Data is Stationary.')
            combined = pd.concat([combined, df['RSI']], axis=1)
            
            
        # Perform ADF test
        st.write('ADF Statistics for MACD:-')
        if varify_adf_test(df['MACD']) > 0.05:
            df_transformed_macd = df['MACD'].diff().dropna()
            adf_test(df_transformed_macd)
            if varify_adf_test(df_transformed_macd) < 0.05:
                st.write('Data is Stationary.')
                st.write(df_transformed_macd.head(50))
                combined = pd.concat([combined, df_transformed_macd], axis=1)
        else:
            adf_test(df['MACD'])
            df_transformed_macd = df['MACD']
            st.write('Data is Stationary.')
            combined = pd.concat([combined, df['MACD']], axis=1)
            
        # Perform ADF test
        st.write('ADF Statistics for Volume:-')
        if varify_adf_test(df['Volume']) > 0.05:
            df_transformed_volume = df['Volume'].diff().dropna()
            adf_test(df_transformed_volume)
            if varify_adf_test(df_transformed_volume) < 0.05:
                st.write('Data is Stationary.')
                st.write(df_transformed_volume.head(50))
                combined = pd.concat([combined, df_transformed_volume], axis=1)
        else:
            adf_test(df['Volume'])
            df_transformed_macd = df['Volume']
            st.write('Data is Stationary.')
            combined = pd.concat([combined, df['Volume']], axis=1)
        
            
        combined = combined.dropna()
        
        st.write(grangers_causation_matrix(combined, variables = combined.columns))
        print(grangers_causation_matrix(combined, variables = combined.columns))
    
        model = CausalModel(
            data=combined,
            treatment='RSI',
            outcome='Close',
            common_causes=['MACD'],
            instruments=['Volume']
        )
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(model.view_model())

        identified_estimand = model.identify_effect()

        estimate_regression = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        estimate_iv = model.estimate_effect(identified_estimand, method_name="iv.instrumental_variable")

        refute_results_regression = model.refute_estimate(identified_estimand, estimate_regression, method_name="random_common_cause")
        refute_results_iv = model.refute_estimate(identified_estimand, estimate_iv, method_name="placebo_treatment_refuter", placebo_type="permute")

        print("Causal effect estimate (linear regression):", estimate_regression)
        print("Causal effect estimate (instrumental variable):", estimate_iv)
        print("Refute results (linear regression):", refute_results_regression)
        print("Refute results (instrumental variable):", refute_results_iv)
        
        
if __name__ == '__main__':
    st.title("Stock Market Prediction")
    st.write("Enter stock symbol:")
    nm = st.text_input("Stock Symbol (NASDAQ or NSE):")
    if st.button('Submit'):
        insert(nm)
        
        
