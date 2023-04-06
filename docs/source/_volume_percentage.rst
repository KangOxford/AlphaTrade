.. _volume_percentage:

Predicting Intraday Trading Volume and Percentages
====================

1. Authors: 
--------------------

Venkatesh Satish, Abhay Saxena, and Max Palmer

2. Affiliation: 
--------------------

Venkatesh Satish and Abhay Saxena are Vice Presidents at FlexTrade Systems in Great Neck, NY. Max Palmer is Head of Trading Analytics and Algorithms at Flex-Trade Systems in Great Neck, NY.

3. Keywords: 
--------------------

intraday volume, volume percentages, trading algorithms, alpha capture, VWAP algorithms

4. Urls: 
--------------------

Paper: None (can be found on https://jot.pm-research.com/content/26/3/24.abstract)

Github code: None

5. Summary:
--------------------

(1): The article discusses techniques for predicting intraday volume and volume percentages in trading. Accurate volume prediction plays an important role in improving trading algorithms' performance, managing risk, and maximizing alpha capture. 

(2): Traditionally, algorithms used historical averages to predict volume during the lifetime of an order, which is not accurate enough. This article provides a well-motivated approach for intraday volume prediction and volume percentage forecasting. 

(3): The authors propose separate models for forecasting raw volume and volume percentages at fixed intervals of time. Their methodology involves various techniques, such as machine learning algorithms and statistical models. 

(4): The authors conduct experiments and evaluate their models on real-world data. The results show that the proposed models outperform traditional methods, especially in cases where high accuracy predictions are required. The performance achieved supports their goals of improving trading algorithms and maximizing alpha capture.

6. Methods: 
--------------------

(1): The proposed methodology involves using separate models for forecasting raw volume and volume percentages at fixed intervals of time. 

(2): Machine learning algorithms such as random forests, support vector regression, and ridge regression, as well as statistical models such as ARMA and SETAR, are used in the models. 

(3): The authors conduct experiments using real-world data to evaluate their models. They compare the results with traditional methods, such as using historical averages to predict volume, and find that the proposed models outperform them in terms of accuracy. 

(4): The intraday volume prediction model is specifically designed to provide reasonable forecasts for all remaining intraday intervals, while the volume percentage forecasting model focuses on predicting only the subsequent interval, which is useful for VWAP algorithms. 

(5): The authors highlight the importance of accurate volume prediction in improving trading algorithms' performance, managing risk, and maximizing alpha capture. The proposed models address the limitations of traditional methods and provide superior results, supporting their goals.

7. Conclusion:
--------------------

(1): This piece of work is significant in improving intraday trading by providing a well-motivated approach for intraday volume prediction and volume percentage forecasting using machine learning algorithms and statistical models. It improves trading algorithms' performance, manages risk, and maximizes alpha capture.

(2): Innovation point: The authors propose a methodology that involves using separate models for forecasting raw volume and volume percentages at fixed intervals of time. They utilize machine learning algorithms and statistical models to improve the accuracy of volume prediction. (3): Performance: The experiments conducted by the authors show that the proposed models outperform traditional methods in terms of accuracy, particularly in high accuracy prediction situations. (4): Workload: The workload in building these models may be high since they require data preprocessing, model selection, parameter tuning, and validation. Nevertheless, the proposed models offer superior results compared to traditional methods, which outweighs the workload.

