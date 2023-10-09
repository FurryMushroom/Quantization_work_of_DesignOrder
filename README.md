# Quantization_work_of_DesignOrder
This repository records the quantization work of LLMs of DesignOrder Company in China mainland.

This repository is contributed by Yixin Jiang, intern of DesignOrder with the responsibility to research on quantization of LLMs, student of Software Engineering School of Tongji University.

I am a fourth grade undergraduate student, with limited knowledge storage and clumsy English expressions.

Thank you for clicking into this page and feel free to correct or give advice to me if there's any deficiency or error!

I've just read some relatively new papers on this topic recently:
* Smoothquant; might be the most practical one, as it focuses on quantizing to int8 format, which has been supported by mainstream nv products. And it has also been encapsulated as a pypi package.

  Transfer quantizaiton difficulty from activations to weights, as the multitude and number of outliers in activations greatly exceed those in weights. Use a small benchmark dataset to get channel-wise activation scales and apply scaling to weights and activations accordingly, do not insert pseudo-quantization nodes between layers in inference. An inspiring idea, but it's doubtful that for different inputs, will the outliers cluster in same channels?
* Olive; might be of great potential, and the experiment results have been reproduced. But it seems to need hardware support to outperform others, E2M1 and E4M3 operations do not seem to be supported by current GPUs.

  Introduces outlier-victim pairs to tackle the problem of quantizing outliers.
* Qlora;
* PEQA;
* RPTQ;
* MoFQ;
* Zeroquant-FP;
* Outlier Supression+;
  
They are on post-training quantization, and mostly devised by Chinese scholars. I wonder if it's I read too few papers or it's just because this area is particularly concerned by Chinese researchers, maybe out of the restriction by US government on Chinese clients purchasing nv GPUs.

