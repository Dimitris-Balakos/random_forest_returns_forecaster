# random_forest_returns_forecaster
In this application a random forest model, which is an ensemble learning model that accounts for non-linearity, will be employed to be trained on the following macroeconomic factors for predicting the SP500 returns:
•	10-Year Treasury Yield (DGS10): Rising interest rates can add pressure to the equity market and they are an important metric for gauging investor risk sentiment.
•	2-Year Treasury Yield (DGS2): Captures expectations for monetary policy and impacts market liquidity. Shifts might signal changes in economic conditions.
•	M2 Money Supply (M2SL): Tracks liquidity conditions; higher M2 might correlate with higher levels of spending which can positively impact equities.
•	Core CPI (CPILFESL): Influences monetary policy and interest rate regime.
•	Industrial Production Index (INDPRO): Tracks industrial output. Strong industrial production might suggest boost conditions for equities.
•	Nonfarm Payrolls (PAYEMS): Measures market labor health which can be positively correlated with equity returns.
•	VIX (VIXCLS): The “fear index” is a key measure of investor sentiment and risk appetite.
•	Consumer Sentiment Index (UMCSENT): Measures consumer confidence which is correlated with spending patterns that drive corporate profits.
•	Crude Oil Prices (DCOILWTICO): Key cost and inflation driver that affects profit margins while it is also a leading indicator of economic activity.
•	Business Inventories (BUSINV): Indicator of supply and demand dynamics. Rising inventories might signal declining consumption.
In aggregate, those macroeconomic factors capture monetary, fiscal, economic, industrial and consumption patterns that can be relevant to equity forecasting. All of the feature variables are differenced to induce stationarity. Additionally, the dependent variable, captured by SP500, is shifted by 1, 2 and 3 months so that the 3 target forecast windows are established. Finally, the dataset is split chronologically to maintain temporal structure with 80% of the available time series used for training and 20% for comparison of the predicted values against the ex-post SP500 returns. A separate random forest model is fitted for each prediction horizon. 

