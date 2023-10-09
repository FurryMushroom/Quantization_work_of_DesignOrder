# Quantization work of DesignOrder
This repository records the quantization work of LLMs of DesignOrder Company in China mainland.

This repository is contributed by Yixin Jiang, intern of DesignOrder with the responsibility to research on quantization of LLMs, student of Software Engineering School of Tongji University.

I am a fourth grade undergraduate student, with limited knowledge storage and clumsy English expressions.

Thank you for clicking into this page and feel free to correct or give advice to me if there's any deficiency or error!

## Comprehension of recently devised approaches
I've just read some relatively new papers on this topic recently:
* Smoothquant; might be the most practical one, as it focuses on quantizing to int8 format, which has been supported by mainstream nv products. And it has also been encapsulated as a pypi package.

  Transfer quantizaiton difficulty from activations to weights, as the multitude and number of outliers in activations greatly exceed those in weights. Use a small benchmark dataset to get channel-wise activation scales and apply scaling to weights and activations accordingly, do not insert pseudo-quantization nodes between layers in inference. An inspiring idea, but it's doubtful that for different inputs, will the outliers cluster in same channels?
* Olive; might be of great potential, and the experiment results about accuracies have been reproduced. But it seems to need hardware support to outperform others, E2M1 and E4M3 operations do not seem to be supported by current GPUs.

  Introduces outlier-victim pairs to tackle the problem of quantizing outliers. Sacrifice the value beside an outlier (in the matrix) to let the outlier have a wider range to represent its value; the victim, who gets sacrificed, is substituted as an identifier. The paper also introduces Abfloat data format innovatively, and proves that E2M1(exponent2, mantissa 1 and sign 1) excels in 4 bit representations and E4M3 excels in 8 bit representations.

* RPTQ;
  Reorder-based PTQ, cluster the channels in the activations and reorganize them for quantization.
  According to the researchers: Firstly, RPTQ is more adept at addressing the challenge of channel difference in activations. By
clustering channels with similar value ranges, it diminishes the influence of both outliers and range
differences. Secondly, RPTQ exhibits memory and computation efficiency, as it only requires
managing quantization parameters for each cluster rather than each individual channel or vector.

  The paper gives a brief introduction to smoothquant to compare with and demonstrate the advantages of RPTQ but I'm suspicious about that. The savings on memory and computation load, theroretically, is not prominent. But clustering channels by K-means rendering scales from channel-wise to cluster-wise intuitively decreases accuracy of representing, expanding quantization errors, often calculated as MSE.

  And what's important is that this hasn't been encapsulated, thus not convenient to use(
* MoFQ;
  Mixture of Formats Quantization, which selects the optimal format on a layer-wise basis.
FP8 has already garnered support from leading hardware vendors, mainly refers to H100 and H800, which offers identical peak performance for FP8 and INT8 operations.
  The paper claims that while FP MAC(multiply-accumlation) generally requires more hardware resources than INT MAC, the resource gap substantially narrows as the bit width decreases,
with FP8 and INT8 costs being notably similar.

  The optimal format is influenced by a combination of various factors,
including the static/dynamic nature of tensors, outlier distribution, and quantization bit-width. For
weight tensors with static distribution, INT quantization outperforms FP quantization at 8-bit but
this advantage diminishes at 4-bit. For activation tensors with dynamic distribution and significant
outliers, FP quantization surpasses INT quantization due to FP can represent large values with lower
precision and small values with higher precision.

  Inspired by the analysis finding of no consistent superior format, the authors propose the approach which selectively determines the optimal format from INT and
FP with the same bit-width on a layer-wise basis. The selection method choosing the format with the minimum quantization error based on metrics such as tensor
MSE, layer output MSE, or model output MSE, proves
effective according to the paper, and also is applied by other researchers. There hasn't been any proof that the smaller quantization MSE is the higher accuracy we get, but it's intuitive.

   Paper draws the conclusion that INT8 is better for weights and FP8 is better
for activation, the accuracy of output activation depends on the impact of weight and input activation being multiplied, so there
is also no consistent superior format for W8A8 quantization. The key idea is to leverage the complementary advantages of both formats in a unified framework, thereby maximizing the potential benefits.

  However, DesignOrder does not possess H100 or more advanced GPUs so far, and I'm also skeptical that MAC between FP and INT retain the efficiency of INT's. 
* Zeroquant-FP;
   This paper claims that FP8 activation consistently outshines its integer (INT8) equivalent, and for weight quantization, FP4 exhibits comparable, if not superior, performance to INT4, simplifying deployment
on FP-supported hardware like H100.

  First it invokes studies that indicate PTQ on 8-bit integer (INT8) weight-only quantization does not compromise the quality of LLMs, and only a
minor accuracy drop is observed with INT4 weight quantization when advanced algorithm such as GPTQ
applied. In studies such as ZeroQuants, SmoothQuant and others, reducing the precision of activation from FP16 to INT8 inevitably results in a decrease in model
quality. Despite the potentially higher computation cost of FP8 compared to INT8 and in light of hardware
support, the improved model quality could make this trade-off worthwhile and merits further exploration. In larger
models, FP8 activation and weight quantization result in negligible degradation. Given the limitations of integer quantization, floating-point methods such as FP8 or FP4, utilizing ExMy
notation, emerge as superior alternatives.

  The actual software implementation of W4A8 in H100 NVIDIA hardware is
that one needs to cast W’s FP4 to match the FP8 precision used in A. The direct method of dequantization
followed by quantization again could potentially have a detrimental effect on inference efficiency, hence the paper constrains S(scaling factor) to be a power of 2 to alleviate.

  At last, it's concluded that FP8 Activation is much better than INT8, FP8 weights rival INT8, while FP4 weights potentially outperform INT4.
  
  However, these are not taken into our account because of lack of hardware support.
* Outlier Supression+;
  Introduces optimal channel-wise shifting and scaling operations, out of the fact that distribution of values or outliers is not only varying between channels, but also asymmetric in every channel. This correspond to mathematic intuitive greatly, and is believed to achieve satisfactory performance in real practice.
  The experiments include comparisons with smoothquant, and outperforms it. What's confusing is that some of the experiments carried out on INT6 format, which hasn't been supported by any hardware as far as I know.
  
They are on post-training quantization of both weights and activations, and mostly devised by Chinese scholars. I wonder if it's I read too few papers or it's just because this area is particularly concerned by Chinese researchers, maybe out of the restriction by US government on Chinese clients purchasing nv GPUs.

The papers or approaches listed above is also introduced in 'A Survey on Model Compression for Large Language Models' by Zhu et al.. 
## Investigations of current available hardware
DesignOrder's main computation hardware is nv A800. Investigate about it will bring benefits in the aspect of quantization, which tightly correlates with hardware.

