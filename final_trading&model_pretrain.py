#!/usr/bin/env python
# coding: utf-8

# ## Pre-train Model

# In[ ]:


#!/usr/bin/env python3

import os
import datetime
import time
import pytz
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi

import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1) ENVIRONMENT SETUP
load_dotenv()

API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_SECRET_API"
BASE_URL = "https://paper-api.alpaca.markets" 

api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version="v2")

SYMBOL = "XOM"
MODEL_PATH = "trading_model.pkl"

# Eastern market hours
MARKET_OPEN  = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)

# Timezone objects
eastern = pytz.timezone("US/Eastern")
utc = pytz.utc

# 2) FETCH 50 DAYS HISTORICAL DATA FOR TRAINING
def fetch_50d_bars(symbol):
    """
    Fetch last 50 days of 1-min bars from Alpaca in UTC,
    convert them to Eastern Time, then filter out non-market hours.
    """
    end_time_utc = datetime.datetime.now(tz=utc)
    start_time_utc = end_time_utc - datetime.timedelta(days=50)

    bars = api.get_bars(
        symbol,
        timeframe="1Min",
        start=start_time_utc.isoformat(),
        end=end_time_utc.isoformat(),
        limit=200000,
        feed="iex"
    )
    records = []
    for bar in bars:
        bar_time_utc = pd.to_datetime(bar.t, unit="s", utc=True)
        bar_time_et = bar_time_utc.astimezone(eastern)

        records.append({
            "timestamp": bar_time_et,  
            "open": bar.o,
            "high": bar.h,
            "low": bar.l,
            "close": bar.c,
            "volume": bar.v
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Filter only market open hours
        df["time"] = df["timestamp"].dt.time
        df = df[(df["time"] >= MARKET_OPEN) & (df["time"] <= MARKET_CLOSE)]
        df.drop(columns=["time"], inplace=True)
    return df

# 3) COMPUTE TECHNICAL INDICATORS
def compute_indicators(df):
    df["SMA_10"] = talib.SMA(df["close"], timeperiod=10)
    df["SMA_30"] = talib.SMA(df["close"], timeperiod=30)
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df.dropna(inplace=True)


# 4) TRAIN & SAVE RANDOM FOREST MODEL

def train_and_save_model(df, model_path=MODEL_PATH):
    """
    Train a Random Forest model using 50 days of data and save it.
    """
    min_ts = df["timestamp"].min()
    boundary_day = min_ts.date() + pd.Timedelta(days=40)  # Use first 40 days for training

    df["date"] = df["timestamp"].dt.date
    train_mask = df["date"] < boundary_day
    test_mask = df["date"] >= boundary_day

    feats = ["SMA_10", "SMA_30", "RSI", "MACD"]
    df["Target"] = df["close"].shift(-1)

    df_train = df[train_mask].dropna(subset=feats + ["Target"])
    df_test = df[test_mask].dropna(subset=feats + ["Target"])

    X_train, y_train = df_train[feats], df_train["Target"]
    X_test, y_test = df_test[feats], df_test["Target"]

    if X_train.empty or X_test.empty:
        print(" Not enough data to train/test ML model.")
        return None

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f" ML Model trained. MSE on test data: {mse:.4f}")

    # Save the model
    joblib.dump(model, model_path)
    print(f" Model saved to {model_path}")
    return model


# 5) MAIN FUNCTION
def main():
    # Fetch historical data
    print("Fetching 50 days of historical market data...")
    df_bars = fetch_50d_bars(SYMBOL)
    
    if df_bars.empty:
        print("No bar data available. Exiting...")
        return

    # Compute indicators
    print("Computing technical indicators...")
    compute_indicators(df_bars)

    # Train & save model
    print("Training and saving ML model...")
    train_and_save_model(df_bars)

if __name__ == "__main__":
    main()


# ## Paper Trading

# In[ ]:


#!/usr/bin/env python3

import os
import re
import datetime
import time
import pytz
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib

import alpaca_trade_api as tradeapi

import torch
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification

# For technical indicators & ML
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

news_sentiments = {} 

# 1) ENVIRONMENT SETUP

load_dotenv()

API_KEY = "PKPNRCT1K71ASC9NS71L"
API_SECRET = "yySLEwa8uATsZrYR3sCk0rHJjDNftDux9yY5yib3"
BASE_URL = "https://paper-api.alpaca.markets" 

api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version="v2")

SYMBOL = "XOM"
MODEL_PATH = "trading_model.pkl"
print("Loading pre-trained model...")
ml_model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Eastern market hours
MARKET_OPEN  = datetime.time(9, 30)  # 9:30 AM ET
MARKET_CLOSE = datetime.time(16, 0)  # 4:00 PM ET

# Timezone objects
eastern = pytz.timezone("US/Eastern")
utc = pytz.utc

# 2) NEWS NLP MODEL
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
nlp_model = BertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
nlp_model.eval()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def analyze_sentiment(text: str) -> float:
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512,
                                   return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = nlp_model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        sentiment_score = torch.argmax(probs, dim=-1).item() + 1  # shift 0..4 → 1..5
    return sentiment_score


# 3) REAL-TIME DATA FETCHING & PROCESSING

def fetch_realtime_data(symbol):
    """
    Fetch the last 180 seconds (3 minutes) of 1-minute bars every 10 seconds.
    Ensure at least 30 bars exist for SMA_30 calculation.
    """
    end_time_utc = datetime.datetime.now(tz=utc)
    start_time_utc = end_time_utc - datetime.timedelta(minutes=30)  # Ensure at least 30 minutes of data

    print(f"Fetching data for {symbol} from {start_time_utc} to {end_time_utc} (UTC)")  

    try:
        bars = api.get_bars(
            symbol,
            timeframe="1Min",
            limit=2000,  # Ensure we get at least 30 minutes of data
            feed="iex"
        )

        if not bars:
            print(" API returned no bars. Possible rate limit issue or invalid API key.")
            return pd.DataFrame()

        records = []
        for bar in bars:
            bar_time_utc = pd.to_datetime(bar.t, unit="s", utc=True)
            bar_time_et = bar_time_utc.astimezone(eastern)

            records.append({
                "timestamp": bar_time_et,
                "close": bar.c,
                "volume": bar.v
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            print("No market data returned. API might be down, or market might be closed.")
            return df  # Return an empty DataFrame

        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Filter only market hours
        df["time"] = df["timestamp"].dt.time
        df = df[(df["time"] >= MARKET_OPEN) & (df["time"] <= MARKET_CLOSE)]
        df.drop(columns=["time"], inplace=True)

        # Debug print after filtering
        print("After filtering:", df.tail(5))

        return df

    except Exception as e:
        print("Error fetching real-time data:", e)
        return pd.DataFrame() 

def fetch_realtime_news(symbol):
    """
    Fetch real-time news and perform sentiment analysis as soon as it is received.
    """
    news_items = []
    try:
        raw_news = api.get_news(symbol=symbol, limit=1)
        for item in raw_news:
            news_time_utc = pd.to_datetime(item.created_at, utc=True)
            news_time_et = news_time_utc.astimezone(eastern)
            combined = f"{item.headline}. {item.summary}. {item.content}".strip()
            sentiment_score = analyze_sentiment(clean_text(combined))
            news_items.append({
                "created_at": news_time_et,
                "combined_text": combined,
                "sentiment_score": sentiment_score
            })
    except Exception as e:
        print("Error fetching news:", e)
    df_news = pd.DataFrame(news_items)
    if not df_news.empty:
        df_news.sort_values("created_at", inplace=True)
        df_news.reset_index(drop=True, inplace=True)
    return df_news


# 4) TECHNICAL INDICATOR ML MODEL

def compute_indicators(df):
    df["SMA_10"] = talib.SMA(df["close"], timeperiod=10)
    df["SMA_30"] = talib.SMA(df["close"], timeperiod=30)
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    df.dropna(inplace=True)


# 5) REAL-TIME TRADING LOGIC

def realtime_trading_logic(symbol, initial_capital=100_000.0):
    global news_sentiments
    
    cash = initial_capital
    shares = 0
    short_pos = 0
    trade_log = []
    last_pnl_check = datetime.datetime.now()
    last_processed_timestamp = None
    last_news_timestamp = None  
    news_trade_count = 0 

    while True:
        current_time = datetime.datetime.now(eastern).time()
        if current_time >= MARKET_OPEN and current_time <= MARKET_CLOSE:
            # Fetch real-time data
            df_bars = fetch_realtime_data(symbol)
            if df_bars.empty:
                print("No bar data, waiting...")
                time.sleep(10)
                continue

            if len(df_bars) < 30: 
                print("Not enough data points for SMA_30 calculation. Waiting for more data...")
                time.sleep(10)
                continue 


            # Compute indicators
            compute_indicators(df_bars)

            # Fetch real-time news
            df_news = fetch_realtime_news(symbol)
            sentiment_score = 3.0
            
            if not df_news.empty:
                latest_news_time = df_news.iloc[-1]["created_at"]
                news_sentiment = df_news.iloc[-1]["sentiment_score"]

                # Store news & track how many trades happened due to it
                if latest_news_time not in news_sentiments:
                    news_sentiments[latest_news_time] = (news_sentiment, 0)

                # Fetch stored sentiment and trade count
                sentiment_score, trade_count = news_sentiments[latest_news_time]

                print(f"News: {latest_news_time} | Sentiment: {sentiment_score} | Trades: {trade_count}")

                # Allow max 2 trades per news event
                if trade_count >= 2:
                    news_sentiments[latest_news_time] = (3.0, trade_count)  # Prevent future trades
                    print(f"Neutralized news sentiment after 2 trades: {news_sentiments[latest_news_time]}")

            print(df_bars.tail())  # See if we are actually getting data

            # Get the latest market data row
            last_row = df_bars.iloc[-1]
            current_timestamp = last_row["timestamp"]

            # **Check if the timestamp has changed (new data arrived)**
            if last_processed_timestamp is None or current_timestamp > last_processed_timestamp:
                print(f"New market data received at {current_timestamp}")
                
                vwap = last_row["VWAP"]

                if trade_count < 2:  # Only trade if <2 trades have occurred for this news
                    if sentiment_score >= 4:
                        signal = "BUY"
                    elif sentiment_score <= 1.5:
                        signal = "SELL"
                    else:
                        signal = "HOLD"

                    if signal in ["BUY", "SELL"]:
                        print(f"Trading on News! Signal: {signal} (Trade {trade_count + 1}/2)")
                        
                        cash, shares, short_pos = execute_trade(last_row["timestamp"], last_row["close"], signal, cash, shares, short_pos, trade_log)

                        trade_count += 1
                        news_sentiments[latest_news_time] = (sentiment_score, trade_count)

                        # If this was the 2nd trade, neutralize sentiment
                        if trade_count == 2:
                            news_sentiments[latest_news_time] = (3.0, trade_count) 
                            print(f"News Sentiment Neutralized After 2 Trades: {news_sentiments}")

                else:
                    signal = combine_signals(sentiment_score, ml_model, last_row, vwap)

                # Execute trade if a valid signal is received
                if signal in ["BUY", "SELL"]:
                    cash, shares, short_pos = execute_trade(
                        last_row["timestamp"], last_row["close"], signal, cash, shares, short_pos, trade_log
                    )
                last_processed_timestamp = current_timestamp
            else:
                print(f"No new market data at {current_timestamp}, skipping duplicate trade execution.")


            # Monitor PNL every 10 minutes
            if (datetime.datetime.now() - last_pnl_check).seconds >= 600:
                current_value = cash + (shares * last_row["close"]) - (short_pos * last_row["close"])
                pnl = current_value - initial_capital
                print(f"Current PNL: ${pnl:.2f}")
                last_pnl_check = datetime.datetime.now()

        else:
            # Overnight logic
            if current_time >= MARKET_CLOSE or current_time < MARKET_OPEN:
                # Fetch and analyze overnight news every 10 minutes
                if (datetime.datetime.now() - last_pnl_check).seconds >= 600:
                    df_news = fetch_realtime_news(symbol)
                    if not df_news.empty:
                        sentiment_score = df_news.iloc[-1]["sentiment_score"]
                        print(f"Overnight sentiment score: {sentiment_score}")
                    last_pnl_check = datetime.datetime.now()

        time.sleep(10)

def combine_signals(sentiment_score, ml_model, row, vwap):
    """
    Determines trading action: BUY, SELL, or HOLD.
    """
    if sentiment_score >= 4:
        print("Sentiment is high, triggering BUY signal!")
        return "BUY"
    elif sentiment_score <= 1.5:
        print("Sentiment is low, triggering SELL signal!")
        return "SELL"
    
    feats = ["SMA_10", "SMA_30", "RSI", "MACD"]
    if any(f not in row or pd.isna(row[f]) for f in feats):
        print(" Not enough data for technical indicators. Holding...")
        return "HOLD"

    current_feats = pd.DataFrame([[row["SMA_10"], row["SMA_30"], row["RSI"], row["MACD"]]], columns=feats)
    predicted = ml_model.predict(current_feats)[0]
    current_price = row["close"]

    if predicted > current_price and vwap > current_price:
        print(" ML model suggests BUY!")
        return "BUY"
    elif predicted < current_price and vwap < current_price:
        print(" ML model suggests SELL!")
        return "SELL"
    
    print("No strong signal. Holding position...")
    return "HOLD"
        
def execute_trade(bar_time, price, signal, cash, shares, short_pos, trade_log, lot_size=10):
    if signal == "BUY":
        print(f"BUY Order Placed at {price} at {bar_time}")
        try:
            api.submit_order(
                symbol=SYMBOL,
                qty=lot_size,
                side="buy",
                type="market",
                time_in_force="gtc"
            )
        except Exception as e:
            print(f"Error placing BUY order: {e}")

        if short_pos > 0:  # Cover short if any
            cost = short_pos * price
            cash -= cost
            trade_log.append((bar_time, "COVER", short_pos, price, cash))
            short_pos = 0
        elif shares == 0:  # Only buy if we don’t have existing shares
            cost = lot_size * price
            cash -= cost
            shares += lot_size
            trade_log.append((bar_time, "BUY", lot_size, price, cash))

    elif signal == "SELL":
        print(f"SELL Order Placed at {price} at {bar_time}")
        try:
            api.submit_order(
                symbol=SYMBOL,
                qty=lot_size,
                side="sell",
                type="market",
                time_in_force="gtc"
            )
        except Exception as e:
            print(f"Error placing SELL order: {e}")

            # Auto-switch to SHORT if not enough shares to sell
            if "insufficient qty" in str(e).lower():
                print(f"Not enough shares to sell ({shares} available). Initiating SHORT instead.")
                try:
                    api.submit_order(
                        symbol=SYMBOL,
                        qty=lot_size,
                        side="sell",  # Shorting in Alpaca is still "sell"
                        type="market",
                        time_in_force="gtc"
                    )
                    trade_log.append((bar_time, "SHORT", lot_size, price, cash))
                    short_pos += lot_size
                except Exception as short_err:
                    print(f"Error placing SHORT order: {short_err}")

        if shares > 0:
            proceeds = shares * price
            cash += proceeds
            trade_log.append((bar_time, "SELL", shares, price, cash))
            shares = 0

    return cash, shares, short_pos

def main():
    realtime_trading_logic(SYMBOL)

if __name__ == "__main__":
    main()

