# Recent Advances in Vision-and-Language Pre-training (VLP)
Maintained by Feilong Chen. Last update on 2023/03/04.
<!-- Last update on 2021/09/14. -->
## Table of Contents
* [Survey](#Survey)
* [Image-based VLP](#Image-based-VLP)
  * [Representation Learning](#representation-learning)
  * [Task-specific](#task-specific)
  * [Other Analysis](#other-analysis)
* [Video-based VLP](#video-based-vlp)
* [Other Transformer-based multimodal networks](#other-transformer-based-multimodal-networks)
* [Other Resources](#other-resources)
<!-- * [Speech-based PTMs](#speech-based-PTMs) -->

# Survey
1. [VLP: A Survey on Vision-Language Pre-training](https://arxiv.org/abs/2202.09061), arXiv 2022

# Image-based VLP

## Representation Learning
0. [Learning Transferable Visual Models From Natural Language Supervision, CLIP](https://arxiv.org/abs/2103.00020), ICML 2021, [[code]](https://github.com/openai/CLIP)

1. [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265), NeurIPS 2019 [[code]](https://github.com/jiasenlu/vilbert_beta)

2. [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490), EMNLP 2019 [[code]](https://github.com/airsplay/lxmert)

3. [VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530), ICLR 2020
 [[code]](https://github.com/jackroos/VL-BERT)

4. [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557), arXiv 2019/08, ACL 2020 [[code]](https://github.com/uclanlp/visualbert)

5. [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/abs/1908.06066), AAAI 2020

6. [Unified Vision-Language Pre-Training for Image Captioning and VQA](https://arxiv.org/pdf/1909.11059.pdf), AAAI 2020, [[code]](https://github.com/LuoweiZhou/VLP), (**VLP**)

7. [UNITER: Learning Universal Image-text Representations](https://arxiv.org/abs/1909.11740), ECCV 2020, [[code]](https://github.com/ChenRocks/UNITER)

8. [Weak Supervision helps Emergence of Word-Object Alignment and improves Vision-Language Tasks](https://arxiv.org/abs/1912.03063), arXiv 2019/12

9. [InterBERT: Vision-and-Language Interaction for Multi-modal Pretraining](https://arxiv.org/abs/2003.13198), arXiv 2020/03

10. [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/pdf/2004.06165.pdf), ECCV 2020, [[code]](https://github.com/microsoft/Oscar)

11. [Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers](https://arxiv.org/abs/2004.00849), arXiv 2020/04

12. [ERNIE-VIL: KNOWLEDGE ENHANCED VISION-LANGUAGE REPRESENTATIONS THROUGH SCENE GRAPH](https://arxiv.org/abs/2006.16934), arXiv 2020/06

13. [DeVLBert: Learning Deconfounded Visio-Linguistic Representations](https://arxiv.org/abs/2008.06884), ACM MM 2020, [[code]](https://github.com/shengyuzhang/DeVLBert)

14. [X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers](https://arxiv.org/abs/2009.11278), EMNLP 2020

15. [SEMVLP: VISION-LANGUAGE PRE-TRAINING BY ALIGNING SEMANTICS AT MULTIPLE LEVELS](https://openreview.net/forum?id=Wg2PSpLZiH), ICLR 2021 submission

16. [CAPT: Contrastive Pre-Training for Learning Denoised Sequence Representations](https://arxiv.org/pdf/2010.06351.pdf), arXiv 2020/10

17. [Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs](https://arxiv.org/pdf/2011.15124.pdf), arXiv 2020/11

18. [LAMP: Label Augmented Multimodal Pretraining](https://arxiv.org/pdf/2012.04446.pdf), arXiv 2020/12

19. [Scheduled Sampling in Vision-Language Pretraining with Decoupled Encoder-Decoder Network](https://arxiv.org/pdf/2101.11562.pdf), AAAI 2021

20. [VinVL: Revisiting Visual Representations in Vision-Language Models](https://arxiv.org/abs/2101.00529), CVPR 2021, [[code]](https://github.com/pzzhang/VinVL)

21. [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/pdf/2102.03334.pdf), ICML 2021, [[code]](https://github.com/dandelin/ViLT)

22. [OPT: Omni-Perception Pre-Trainer for Cross-Modal Understanding and Generation](https://arxiv.org/abs/2107.00249.pdf), arXiv 2021

23. [UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/abs/2012.15409), ACL 2021, [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/UNIMO) 

24. [How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383), arXiv 2021, [[code]](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-Pretrain) 

25. [Unifying Vision-and-Language Tasks via Text Generation](https://arxiv.org/abs/2102.02779), ICML 2021, [[code]](https://github.com/j-min/VL-T5#unifying-vision-and-language-tasks-via-text-generation)
26. [Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs](https://arxiv.org/abs/2011.15124), ACL 2021, [[code]](https://github.com/e-bug/volta)

27. [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/abs/2108.10904), arXiv 2021

28. [VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358), arXiv 2021, [[code]](https://github.com/microsoft/unilm/tree/master/vlmo)

29. [Kaleido-BERT: Vision-Language Pre-training on Fashion Domain](https://arxiv.org/abs/2103.16110), CVPR2021, [[code]](http://dpfan.net/Kaleido-BERT)

30. [Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts](https://arxiv.org/abs/2111.08276), ICML 2022, [[code]](https://github.com/zengyan-97/X-VLM)

32. [Vision-Language Pre-Training with Triple Contrastive Learning](https://arxiv.org/abs/2202.10401), CVPR 2022, [[code]](https://github.com/uta-smile/TCL)

33. [Unpaired Vision-Language Pre-training via Cross-Modal CutMix](https://arxiv.org/abs/2206.08919), ICML 2022

34. [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086), ICML 22, [[code]](https://github.com/salesforce/BLIP)

35. [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052), ICML 22, [[code]](https://github.com/OFA-Sys/OFA)

36. [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100), arXiv 2022, [[code]](https://github.com/microsoft/GenerativeImage2Text)

37. [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917), arXiv 2022, [[code]](https://github.com/lucidrains/CoCa-pytorch)

38. [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442), arXiv 2022, [[code]](https://aka.ms/beit-3)

39. [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794), arXiv 2022

40. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597.pdf), arXiv 2023

41. [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/pdf/2302.14045.pdf), arXiv 2023, [[code]](https://github.com/microsoft/unilm)

42. [Unifying Vision-Language Representation Space with Single-tower Transformer](https://arxiv.org/abs/2211.11153), AAAI 2023

43. [Improved baselines for vision-language pre-training](https://arxiv.org/abs/2305.08675), arXiv 2023

## Task-specific

### Image Caption

1. **Image captioning**: [XGPT: Cross-modal Generative Pre-Training for Image Captioning](https://arxiv.org/abs/2003.01473), arXiv 2020/03

### VQA

1. **VQA**: [Fusion of Detected Objects in Text for Visual Question Answering](https://arxiv.org/abs/1908.05054), EMNLP 2019, [[code]](https://github.com/google-research/language/tree/master/language/question_answering/b2t2), (**B2T2**)

2. **TextVQA**: [Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA](https://arxiv.org/abs/1911.06258), CVPR 2020, [[code]](https://github.com/ronghanghu/pythia/tree/project/m4c/projects/M4C), (**M4C**)

3. **Chart VQA**: [STL-CQA: Structure-based Transformers with Localization and Encoding for Chart Question Answering](https://www.aclweb.org/anthology/2020.emnlp-main.264.pdf), EMNLP 2020.

4. **Visual Question Generation**: [BERT Can See Out of the Box: On the Cross-modal Transferability of Text Representations](https://arxiv.org/abs/2002.10832), arXiv 2020/02

5. **TextVQA**: [TAG: Boosting Text-VQA via Text-aware Visual Question-answer Generation](https://arxiv.org/abs/2208.01813), arXiv 2022, [[code]](https://github.com/HenryJunW/TAG), (**TAG**)

### Visual Dialog

1. **VisDial**: [VD-BERT: A Unified Vision and Dialog Transformer with BERT](https://arxiv.org/abs/2004.13278), EMNLP 2020 [[code]](https://github.com/salesforce/VD-BERT), (**VD-BERT**)

2. **VisDial**: [Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline](https://arxiv.org/abs/1912.02379), ECCV 2020 [[code]](https://github.com/vmurahari3/visdial-bert), (**VisDial-BERT**)

3. **VisDial**: [UTC: A Unified Transformer with Inter-Task Contrastive Learning for Visual Dialog](https://arxiv.org/abs/2205.00423), CVPR 2022

### Text-Image Retrieval

1. **Text-image retrieval**: [ImageBERT: Cross-Modal Pre-training with Large-scale Weak-supervised Image-text Data](https://arxiv.org/abs/2001.07966), arXiv 2020/01

2. **Text-image retrieval**: [CROSS-PROBE BERT FOR EFFICIENT AND EFFECTIVE CROSS-MODAL SEARCH](https://openreview.net/forum?id=bW9SYKHcZiz), ICLR 2021 submission. 

3. **Text-image retrieval**: [Learning Relation Alignment for Calibrated Cross-modal Retrieval](https://github.com/lancopku/IAIS), ACL 2021.

4. **Text-image retrieval**: [Dynamic Contrastive Distillation for Image-Text Retrieval](https://arxiv.org/abs/2207.01426), arXiv 2022/07.

5. **Text-image retrieval**: [Where Does the Performance Improvement Come From? - A Reproducibility Concern about Image-Text Retrieval](https://arxiv.org/abs/2203.03853), SIGIR 2022.

### Visual Language Navigation

1. **VLN**: [Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training](https://arxiv.org/abs/2002.10638), CVPR 2020, [[code]](https://github.com/weituo12321/PREVALENT), (**PREVALENT**)

### Visual Machine Reading Comprehension
1. **VisualMRC**: [VisualMRC: Machine Reading Comprehension on Document Images](https://arxiv.org/abs/2101.11272), AAAI 2021, (**LayoutT5, LayoutBART**)

### Other Tasks

1. **Visual Relationship Detection**: [Visual Relationship Detection With Visual-Linguistic Knowledge From Multimodal Representations](https://ieeexplore.ieee.org/document/9387302), 	IEEE Access 2021


## Other Analysis

1. **Multi-task Learning**, [12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315), CVPR 2020, [[code]](https://github.com/facebookresearch/vilbert-multi-task) 

2. **Multi-task Learning**, [Unifying Vision-and-Language Tasks via Text Generation](https://arxiv.org/abs/2102.02779), arXiv 2021/02

3. **Social Bias in VL Embedding**, [Measuring Social Biases in Grounded Vision and Language Embeddings](https://arxiv.org/abs/2002.08911), arXiv 2020/02, [[code]](https://github.com/candacelax/bias-in-vision-and-language)

4. **In-depth Analysis**, [Are we pretraining it right? Digging deeper into visio-linguistic pretraining](https://arxiv.org/abs/2004.08744),

5. **In-depth Analysis**, [Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models](https://arxiv.org/abs/2005.07310), ECCV 2020 Spotlight

6. **In-depth Analysis**, [A Closer Look at the Robustness of Vision-and-Language Pre-trained Models](https://arxiv.org/abs/2012.08673), arXiv 2020/12

7. **Adversarial Training**, [Large-Scale Adversarial Training for Vision-and-Language Representation Learning](https://arxiv.org/abs/2006.06195), NeurIPS 2020 Spotlight

8. **Adaptive Analysis**, [Adaptive Transformers for Learning Multimodal Representations](https://arxiv.org/abs/2005.07486), ACL SRW 2020


9. **Neural Architecture Search**, [Deep Multimodal Neural Architecture Search](https://arxiv.org/abs/2004.12070), arXiv 2020/04

10. **Dataset perspective**, [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918), arXiv 2021/02
  
  
# Video-based VLP

1. [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766), ICCV 2019

2. [Learning Video Representations Using Contrastive Bidirectional Transformers](https://arxiv.org/abs/1906.05743), arXiv 2019/06, (**CBT**)

3. [M-BERT: Injecting Multimodal Information in the BERT Structure](https://arxiv.org/abs/1908.05787), arXiv 2019/08

4. [BERT for Large-scale Video Segment Classification with Test-time Augmentation](https://arxiv.org/abs/1912.01127), 	ICCV 2019 YouTube8M workshop, [[code]](https://github.com/hughshaoqz/3rd-Youtube8M-TM)

5. [Bridging Text and Video: A Universal Multimodal Transformer for Video-Audio Scene-Aware Dialog](https://arxiv.org/abs/2002.00163), AAAI2020 DSTC8 workshop

6. [Learning Spatiotemporal Features via Video and Text Pair Discrimination](https://arxiv.org/abs/2001.05691), arXiv 2020/01, (**CPD**), [[code]](https://github.com/MCG-NJU/CPD-Video)

7. [UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation](https://arxiv.org/abs/2002.06353), arXiv 2020/02

8. [ActBERT: Learning Global-Local Video-Text Representations](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.html), CVPR 2020

9. [HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training](https://arxiv.org/abs/2005.00200), EMNLP 2020

10. [Video-Grounded Dialogues with Pretrained Generation Language Models](https://arxiv.org/abs/2006.15319), ACL 2020

11. [Auto-captions on GIF: A Large-scale Video-sentence Dataset for Vision-language Pre-training](https://arxiv.org/abs/2007.02375), arXiv 2020/07

12. [Multimodal Pretraining for Dense Video Captioning](https://arxiv.org/pdf/2011.11760.pdf), arXiv 2020/11

13. [PARAMETER EFFICIENT MULTIMODAL TRANSFORMERS FOR VIDEO REPRESENTATION LEARNING](https://arxiv.org/pdf/2012.04124.pdf), arXiv 2020/12

14. [Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling](https://arxiv.org/pdf/2102.06183.pdf), CVPR 2021


<!-- 
# Speech-based PTMs

1. [Towards Transfer Learning for End-to-End Speech Synthesis from Deep Pre-Trained Language Models](https://arxiv.org/abs/1906.07307), arXiv 2019/06

2. [Understanding Semantics from Speech Through Pre-training](https://arxiv.org/abs/1909.10924), arXiv 2019/09

3. [SpeechBERT: Cross-Modal Pre-trained Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559), arXiv 2019/10

4. [vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](https://arxiv.org/abs/1910.05453),  arXiv 2019/10

5. [Effectiveness of self-supervised pre-training for speech recognition](https://arxiv.org/abs/1911.03912),  arXiv 2019/11 -->

# Other Transformer-based multimodal networks

1. [Multi-Modality Cross Attention Network for Image and Sentence Matching](http://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.html), ICCV 2020

2. [MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning](https://arxiv.org/abs/2005.05402), ACL 2020

3. [History for Visual Dialog: Do we really need it?](https://arxiv.org/pdf/2005.07493.pdf), ACL 2020

4. [Cross-Modality Relevance for Reasoning on Language and Vision](https://arxiv.org/abs/2005.06035), ACL 2020



# Other Resources

* Two recent surveys on pretrained language models
  * [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271), arXiv 2020/03
  * [A Survey on Contextual Embeddings](https://arxiv.org/abs/2003.07278), arXiv 2020/03
* Other surveys about multimodal research
  * [Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://arxiv.org/abs/1907.09358), arXiv 2019
  * [Deep Multimodal Representation Learning: A Survey](https://ieeexplore.ieee.org/abstract/document/8715409), arXiv 2019 
  * [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), TPAMI 2018
  * [A Comprehensive Survey of Deep Learning for Image Captioning](https://arxiv.org/abs/1810.04020), ACM Computing Surveys 2018
* Other repositories of relevant reading list
  * [Pre-trained Languge Model Papers from THU-NLP](https://github.com/thunlp/PLMpapers)
  * [BERT-related Papers](https://github.com/tomohideshibata/BERT-related-papers)
  * [Reading List for Topics in Multimodal Machine Learning](https://github.com/pliang279/awesome-multimodal-ml)
  * [A repository of vision and language papers](https://github.com/sangminwoo/awesome-vision-and-language-papers)
  * [Recent Advances in Vision and Language PreTrained Models (VL-PTMs)](https://github.com/yuewang-cuhk/awesome-vision-language-pretraining-papers)
* Simple Survey on VLP
  * VLP Survey on Representation Learning, Feilong Chen, [BaiduYun](https://pan.baidu.com/s/1mb3u6w-0fFim2zmxUa0e_w) password:bujb 
  * VLP Survey on Multimodal Retrieval, Duoduo Feng, [BaiduYun](https://pan.baidu.com/s/1pH5Ath3y0C3n_wAa55xdgA), password:xobv

