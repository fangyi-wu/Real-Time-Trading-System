# **Real-Time Trading System**
🚀 *An automated trading system integrating real-time market data, machine learning, and sentiment analysis for optimized trade execution.*

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Trading Strategy](#trading-strategy)
- [Backtesting System](#backtesting-system)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results & Performance](#results--performance)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## **Introduction**
The **Real-Time Trading System** is designed to automate stock trading decisions for **ExxonMobil (XOM)** using **Alpaca's API** in a **paper trading environment**. By leveraging **machine learning models, technical indicators, and sentiment analysis**, this system optimizes trade execution and enhances decision-making efficiency. 

This project follows a **structured, data-driven approach**:
- **Real-time stock price retrieval** from **Alpaca API**
- **Integration of technical indicators** (SMA, RSI, MACD, VWAP)
- **Machine learning-based price predictions** (Random Forest Regressor)
- **News sentiment analysis** using **BERT NLP models**
- **Automated trade execution** with risk management filters
- **Backtesting system** for strategy optimization

---

## **Features**
✅ **Real-time data processing** – Fetches stock prices every **10 seconds**  
✅ **Algorithmic trading strategies** – Uses SMA, RSI, MACD, and VWAP  
✅ **Machine learning-powered predictions** – Implements **Random Forest** for price forecasting  
✅ **News sentiment integration** – BERT-based NLP sentiment analysis to trigger trades  
✅ **Automated trade execution** – Decision hierarchy ensures quality signal filtering  
✅ **Backtesting support** – Evaluates past market data for strategy validation  
✅ **Paper trading environment** – Ensures no real money is at risk  

---

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/fangyi-wu/Real-Time-Trading-System.git
cd Real-Time-Trading-System
