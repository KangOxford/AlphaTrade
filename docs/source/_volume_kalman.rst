.. _volume_kalman:

Forecasting Intraday Trading Volume: A Kalman Filter Approach
====================

1. Authors:
--------------------

Zhifeng Zhang, Zhenyu Chen, Yan Chen, Xiaodong Liu

2. Affiliation:
--------------------

None provided

3. Keywords:
--------------------

algorithmic trading, EM, intraday trading volume, Kalman filter, Lasso, VWAP

4. Urls:
--------------------

Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3101695

Github: None

5. Summary:
--------------------

(1): This paper focuses on forecasting intraday trading volume, which is important for algorithmic trading purposes.



(2): Previous methods include rolling means (RM) and component multiplicative error model (CMEM), but they fail to capture the intraday regularities in trading volume. The proposed approach is motivated by the need for a more accurate and effective model.

(3): The paper proposes a state-space model based on the Kalman filter to forecast intraday trading volume and the EM algorithm for parameter estimation. A Lasso regularization technique is also applied to handle outliers in real-time market data.

(4): The proposed model outperforms RM and CMEM by 64% and 29% respectively in terms of volume prediction and by 15% and 9% respectively in Volume Weighted Average Price (VWAP) trading. These performance results support the goal of achieving more accurate and effective intraday trading volume forecasting.

6. Conclusion:
--------------------

(1): The significance of this piece of work lies in proposing a new methodology for intraday trading volume forecasting that outperforms previous approaches and can contribute to more accurate and effective algorithmic trading strategies.



(2): Innovation point: The proposed approach is based on a state-space model using the Kalman filter algorithm and includes a Lasso regularization technique for handling outliers in real-time market data. (3): Performance: The proposed model outperforms previous methods by a significant margin in terms of volume prediction and VWAP trading replication. (4): Workload: The proposed methodology is computationally efficient and reduces the forecast complexity. However, the study is limited to a specific dataset and further testing on a broader set of stocks and markets is needed.

