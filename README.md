This project was designed as a proof of concept and training ground within technologies to analyze, and forecast its future using different machine learning models. 
  * Linear Regression
  * Random Forrest (not completed)
  * Arima (not started)

I set out to answer a few questions:
  * Are there certain industries, and stocks that have high correlation that is significant to the index?
  *   For this, I used QQQ as the Nasdaq-100 index
  * Which individual ticker maintains a high correlation to the Nasdaq-100 index?
  * Can I predict 2024 using different training sets?
  * Who has better price control indicating stability?
  * Can i obtain 70% forecast / prediction accuracy?

Tasks:
  * Extract data from yFinance library
  * Load data into local SQL Server for perm storage (offline use)
  * Load data into Linear Regression Model
  *   Version 1.1.1 = 2010 - 2023 to predict 2024
  *   Version 1.1.2 = 2023 to predict 2024
  * Report findings of market and models using:
  *   Tableau
  *   PowerBI

Assumed extent of available data:
  * Date
  * Open
  * Close
  * High
  * Low
  * Volume

