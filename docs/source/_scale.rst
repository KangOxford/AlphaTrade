.. _scale:

SCALEFORMER: Iterative multi-scale refining transformers for time series forecasting, February 2023
====================

1. Authors:
--------------------

Mohammad Amin Shabani, Amir Abdi, Lili Meng, Tristan Sylvain

2. Affiliation:
--------------------

Mohammad Amin Shabani: Simon Fraser University, Canada & Borealis AI, Canada

Amir Abdi, Lili Meng, Tristan Sylvain: Borealis AI, Canada



3. Keywords:
--------------------

time series forecasting, transformers, multi-scale framework, iterative refining, shared weights



4. Urls:
--------------------

https://arxiv.org/abs/2206.04038, Github: https://github.com/BorealisAI/scaleformer



5. Summary:
--------------------



(1): This paper focuses on time series forecasting and achieving scale awareness in transformer-based models.



(2): The past approaches mainly focused on mitigating the standard quadratic complexity in time and space, rather than explicit scale-awareness. The essential cross-scale feature relationships were often learned implicitly. With Scaleformer, the time series forecasts are iteratively refined at successive time-steps, allowing the model to better capture the inter-dependencies and specificities of each scale, making transformers more scale-aware.



(3): The proposed methodology is a general multi-scale framework that can be applied to the state-of-the-art transformer-based time series forecasting models. It introduces iterative refinement of a forecasted time series at multiple scales with shared weights, introducing architecture adaptations, and a specially-designed normalization scheme.



(4): The proposed improvements outperform their corresponding baseline counterparts, achieving significant performance improvements, from 5.5% to 38.5% across datasets and transformer architectures, with minimal additional computational overhead as demonstrated by detailed ablation studies. The code is publicly available.

6. Methods:
--------------------

(1): The main idea of this paper is to develop a multi-scale refinement framework, called Scaleformer, for time series forecasting using transformer-based models, which allows the model to better capture the essential cross-scale feature relationships explicitly.



(2): The proposed methodology employs an iterative refinement process of forecasted time series at multiple scales using shared weights, which is designed to adapt to various transformer-based time series forecasting models with minimal computational overhead. The architecture adaptations also introduce a specially-designed normalization scheme to enhance the model's effectiveness in capturing cross-scale relationships.



(3): The authors evaluated the performance improvements of the proposed framework over several benchmark datasets using traditional evaluation metrics such as mean absolute error (MAE) and mean squared error (MSE). They conducted a comprehensive ablation study to assess the effectiveness of individual components and hyperparameters of the proposed model. The experimental results demonstrate the superiority of the proposed approach in terms of accuracy and robustness compared to corresponding baselines. The code for the proposed Scaleformer framework is publicly available.

7. Conclusion: 
--------------------

(1): This work is significant as it proposes a novel multi-scale refinement framework, called Scaleformer, for time series forecasting using transformer-based models, explicitly capturing essential cross-scale feature relationships that were previously learned implicitly.



(2): Innovation point: The proposed approach introduces iterative refinement of a forecasted time series at multiple scales with shared weights, an architecture adaptation, and a specially-designed normalization scheme for enhanced performance. (3): Performance: The proposed improvements outperform their corresponding baseline counterparts, achieving significant performance improvements, from 5.5% to 38.5% across datasets and transformer architectures, with minimal additional computational overhead. (4): Workload: The proposed methodology is a general multi-scale framework that can be applied to the state-of-the-art transformer-based time series forecasting models, and the code is publicly available for reproducibility.

