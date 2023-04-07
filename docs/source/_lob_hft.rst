.. _lob_hft:

Axial-LOB: High-Frequency Trading with Axial Attention, January 2023
====================

                 

                 

1. Authors:
--------------------

Damian Kisiel, Denise Gorse





2. Affiliation:
--------------------

Department of Computer Science, University College London





3. Keywords:
--------------------

Deep Learning, Axial Attention, High-Frequency Trading, Limit Order Book Data





4. Urls:
--------------------

https://arxiv.org/abs/2212.01807v1, Github: None





5. Summary:
--------------------

(1): This article is researching the prediction of stock price movements from limit order book (LOB) data.



(2): The past methods mostly relied on deep convolutional neural networks which were limited to local interactions, potentially missing out on long-range dependencies. Recent studies addressed this problem by using additional recurrent or attention layers, but they increased computational complexity. The approach of this paper is the use of axial attention layers to construct feature maps that incorporate global interactions while significantly reducing the size of the parameter space. Axial-LOB does not rely on hand-crafted convolutional kernels and has stable performance under input permutations.



(3): The research methodology proposed in this paper is a novel fully-attentional deep learning architecture for predicting price movements of stocks from LOB data. Axial-LOB uses axial attention layers that factorize the standard 2D attention mechanism into two 1D self-attention blocks to recover the global receptive field in a computationally efficient manner. Additionally, gated positional embeddings are used within the attention mechanisms to utilize and control position-dependent interactions.



(4): The effectiveness of Axial-LOB is demonstrated on a large benchmark dataset containing time series representations of millions of high-frequency trading events, achieving a new state-of-the-art directional classification performance at all tested prediction horizons. The model has lower complexity and demonstrates stable performance under permutations of the input data, supporting the goals of the approach.

6. Methods: 
--------------------

(1): The proposed methodology is a fully-attentional deep learning architecture for predicting stock price movements using limit order book (LOB) data. 

(2): The approach uses axial attention layers to construct feature maps that incorporate global interactions while significantly reducing the size of the parameter space. The axial attention layers factorize the standard 2D attention mechanism into two 1D self-attention blocks to recover the global receptive field in a computationally efficient manner. 

(3): Gated positional embeddings are used within the attention mechanisms to utilize and control position-dependent interactions. The model does not rely on hand-crafted convolutional kernels and has stable performance under input permutations. 

(4): The effectiveness of the proposed model was demonstrated using a large benchmark dataset containing time series representations of millions of high-frequency trading events, achieving a new state-of-the-art directional classification performance at all tested prediction horizons. Training was done via mini-batch stochastic gradient descent (SGD) by minimizing the cross-entropy loss between the predicted class probabilities and the ground truth label.

7. Conclusion: 
--------------------

(1): The significance of this work is the proposal of a novel fully-attentional deep learning architecture for predicting stock price movements from limit order book (LOB) data. The approach uses axial attention layers to factorize the standard 2D attention mechanism, providing a computationally efficient method to incorporate global interactions without the need for hand-crafted convolutional kernels. 

(2): Innovation point: The use of axial attention layers provides a more efficient approach to incorporating global interactions in LOB data prediction without the need for hand-crafted convolutional kernels. (3): Performance: The proposed model achieves a new state-of-the-art directional classification performance in all tested prediction horizons on the benchmark dataset. (4): Workload: The model has lower complexity compared to previous methods, supporting its practical use.

