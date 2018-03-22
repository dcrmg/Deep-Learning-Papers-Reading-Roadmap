# Deep Learning Papers Reading Roadmap

>If you are a newcomer to the Deep Learning area, the first question you may have is "Which paper should I start reading from?"

>Here is a reading roadmap of Deep Learning papers!

The roadmap is constructed in accordance with the following four guidelines:

- From outline to detail
- From old to state-of-the-art
- from generic to specific areas
- focus on state-of-the-art

You will find many papers that are quite new but really worth reading.

I would continue adding papers to this roadmap.


---------------------------------------

# 1 Deep Learning History and Basics

## 1.0 Book

**[0]** Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. "**Deep learning**." An MIT Press book. (2015). [[html]](http://www.deeplearningbook.org/) **(Deep Learning Bible, you can read this book while reading following papers.)** :star::star::star::star::star:

## 1.1 Survey

**[1]** LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "**Deep learning**." Nature 521.7553 (2015): 436-444. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) **(Three Giants' Survey)** :star::star::star::star::star:

## 1.2 Deep Belief Network(DBN)(Milestone of Deep Learning Eve)

**[2]** Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "**A fast learning algorithm for deep belief nets**." Neural computation 18.7 (2006): 1527-1554. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)**(Deep Learning Eve)** :star::star::star:

**[3]** Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "**Reducing the dimensionality of data with neural networks**." Science 313.5786 (2006): 504-507. [[pdf]](http://www.cs.toronto.edu/~hinton/science.pdf) **(Milestone, Show the promise of deep learning)** :star::star::star:

## 1.3 ImageNet Evolution（Deep Learning broke out from here）

**[4]** Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) **(AlexNet, Deep Learning Breakthrough)** :star::star::star::star::star:

**[5]** Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**." arXiv preprint arXiv:1409.1556 (2014). [[pdf]](https://arxiv.org/pdf/1409.1556.pdf) **(VGGNet,Neural Networks become very deep!)** :star::star::star:

**[6]** Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) **(GoogLeNet)** :star::star::star:

**[7]** He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015). [[pdf]](https://arxiv.org/pdf/1512.03385.pdf) **(ResNet,Very very deep networks, CVPR best paper)** :star::star::star::star::star:

## 1.4 Speech Recognition Evolution

**[8]** Hinton, Geoffrey, et al. "**Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups**." IEEE Signal Processing Magazine 29.6 (2012): 82-97. [[pdf]](http://cs224d.stanford.edu/papers/maas_paper.pdf) **(Breakthrough in speech recognition)**:star::star::star::star:

**[9]** Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "**Speech recognition with deep recurrent neural networks**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1303.5778.pdf) **(RNN)**:star::star::star:

**[10]** Graves, Alex, and Navdeep Jaitly. "**Towards End-To-End Speech Recognition with Recurrent Neural Networks**." ICML. Vol. 14. 2014. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf):star::star::star:

**[11]** Sak, Haşim, et al. "**Fast and accurate recurrent neural network acoustic models for speech recognition**." arXiv preprint arXiv:1507.06947 (2015). [[pdf]](http://arxiv.org/pdf/1507.06947) **(Google Speech Recognition System)** :star::star::star:

**[12]** Amodei, Dario, et al. "**Deep speech 2: End-to-end speech recognition in english and mandarin**." arXiv preprint arXiv:1512.02595 (2015). [[pdf]](https://arxiv.org/pdf/1512.02595.pdf) **(Baidu Speech Recognition System)** :star::star::star::star:

**[13]** W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig "**Achieving Human Parity in Conversational Speech Recognition**." arXiv preprint arXiv:1610.05256 (2016). [[pdf]](https://arxiv.org/pdf/1610.05256v1) **(State-of-the-art in speech recognition, Microsoft)** :star::star::star::star:

>After reading above papers, you will have a basic understanding of the Deep Learning history, the basic architectures of Deep Learning model(including CNN, RNN, LSTM) and how deep learning can be applied to image and speech recognition issues. The following papers will take you in-depth understanding of the Deep Learning method, Deep Learning in different areas of application and the frontiers. I suggest that you can choose the following papers based on your interests and research direction.

#2 Deep Learning Method

## 2.1 Model

**[14]** Hinton, Geoffrey E., et al. "**Improving neural networks by preventing co-adaptation of feature detectors**." arXiv preprint arXiv:1207.0580 (2012). [[pdf]](https://arxiv.org/pdf/1207.0580.pdf) **(Dropout)** :star::star::star:

**[15]** Srivastava, Nitish, et al. "**Dropout: a simple way to prevent neural networks from overfitting**." Journal of Machine Learning Research 15.1 (2014): 1929-1958. [[pdf]](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) :star::star::star:

**[16]** Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." arXiv preprint arXiv:1502.03167 (2015). [[pdf]](http://arxiv.org/pdf/1502.03167) **(An outstanding Work in 2015)** :star::star::star::star:

**[17]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "**Layer normalization**." arXiv preprint arXiv:1607.06450 (2016). [[pdf]](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote) **(Update of Batch Normalization)** :star::star::star::star:

**[18]** Courbariaux, Matthieu, et al. "**Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**." [[pdf]](https://pdfs.semanticscholar.org/f832/b16cb367802609d91d400085eb87d630212a.pdf) **(New Model,Fast)**  :star::star::star:

**[19]** Jaderberg, Max, et al. "**Decoupled neural interfaces using synthetic gradients**." arXiv preprint arXiv:1608.05343 (2016). [[pdf]](https://arxiv.org/pdf/1608.05343) **(Innovation of Training Method,Amazing Work)** :star::star::star::star::star:

**[20]** Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [[pdf]](https://arxiv.org/abs/1511.05641) **(Modify previously trained network to reduce training epochs)** :star::star::star:

**[21]** Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [[pdf]](https://arxiv.org/abs/1603.01670) **(Modify previously trained network to reduce training epochs)** :star::star::star:

## 2.2 Optimization

**[22]** Sutskever, Ilya, et al. "**On the importance of initialization and momentum in deep learning**." ICML (3) 28 (2013): 1139-1147. [[pdf]](http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf) **(Momentum optimizer)** :star::star:

**[23]** Kingma, Diederik, and Jimmy Ba. "**Adam: A method for stochastic optimization**." arXiv preprint arXiv:1412.6980 (2014). [[pdf]](http://arxiv.org/pdf/1412.6980) **(Maybe used most often currently)** :star::star::star:

**[24]** Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." arXiv preprint arXiv:1606.04474 (2016). [[pdf]](https://arxiv.org/pdf/1606.04474) **(Neural Optimizer,Amazing Work)** :star::star::star::star::star:

**[25]** Han, Song, Huizi Mao, and William J. Dally. "**Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**." CoRR, abs/1510.00149 2 (2015). [[pdf]](https://pdfs.semanticscholar.org/5b6c/9dda1d88095fa4aac1507348e498a1f2e863.pdf) **(ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup)** :star::star::star::star::star:

**[26]** Iandola, Forrest N., et al. "**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**." arXiv preprint arXiv:1602.07360 (2016). [[pdf]](http://arxiv.org/pdf/1602.07360) **(Also a new direction to optimize NN,DeePhi Tech Startup)** :star::star::star::star:

## 2.3 Unsupervised Learning / Deep Generative Model

**[27]** Le, Quoc V. "**Building high-level features using large scale unsupervised learning**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1112.6209.pdf&embed) **(Milestone, Andrew Ng, Google Brain Project, Cat)** :star::star::star::star:


**[28]** Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013). [[pdf]](http://arxiv.org/pdf/1312.6114) **(VAE)** :star::star::star::star:

**[29]** Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN,super cool idea)** :star::star::star::star::star:

**[30]** Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](http://arxiv.org/pdf/1511.06434) **(DCGAN)** :star::star::star::star:

**[31]** Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) **(VAE with attention, outstanding work)** :star::star::star::star::star:

**[32]** Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](http://arxiv.org/pdf/1601.06759) **(PixelRNN)** :star::star::star::star:

**[33]** Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)** :star::star::star::star:

## 2.4 RNN / Sequence-to-Sequence Model

**[34]** Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](http://arxiv.org/pdf/1308.0850) **(LSTM, very nice generating result, show the power of RNN)** :star::star::star::star:

**[35]** Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](http://arxiv.org/pdf/1406.1078) **(First Seq-to-Seq Paper)** :star::star::star::star:

**[36]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](https://arxiv.org/pdf/1409.3215.pdf) **(Outstanding Work)** :star::star::star::star::star:

**[37]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf) :star::star::star::star:

**[38]** Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) **(Seq-to-Seq on Chatbot)** :star::star::star:

## 2.5 Neural Turing Machine

**[39]** Graves, Alex, Greg Wayne, and Ivo Danihelka. "**Neural turing machines**." arXiv preprint arXiv:1410.5401 (2014). [[pdf]](http://arxiv.org/pdf/1410.5401.pdf) **(Basic Prototype of Future Computer)** :star::star::star::star::star:

**[40]** Zaremba, Wojciech, and Ilya Sutskever. "**Reinforcement learning neural Turing machines**." arXiv preprint arXiv:1505.00521 362 (2015). [[pdf]](https://pdfs.semanticscholar.org/f10e/071292d593fef939e6ef4a59baf0bb3a6c2b.pdf) :star::star::star:

**[41]** Weston, Jason, Sumit Chopra, and Antoine Bordes. "**Memory networks**." arXiv preprint arXiv:1410.3916 (2014). [[pdf]](http://arxiv.org/pdf/1410.3916) :star::star::star:


**[42]** Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. "**End-to-end memory networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf) :star::star::star::star:

**[43]** Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. "**Pointer networks**." Advances in Neural Information Processing Systems. 2015. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf) :star::star::star::star:

**[44]** Graves, Alex, et al. "**Hybrid computing using a neural network with dynamic external memory**." Nature (2016). [[pdf]](https://www.dropbox.com/s/0a40xi702grx3dq/2016-graves.pdf) **(Milestone,combine above papers' ideas)** :star::star::star::star::star:

## 2.6 Deep Reinforcement Learning

**[45]** Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." arXiv preprint arXiv:1312.5602 (2013). [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)) **(First Paper named deep reinforcement learning)** :star::star::star::star:

**[46]** Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**." Nature 518.7540 (2015): 529-533. [[pdf]](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) **(Milestone)** :star::star::star::star::star:

**[47]** Wang, Ziyu, Nando de Freitas, and Marc Lanctot. "**Dueling network architectures for deep reinforcement learning**." arXiv preprint arXiv:1511.06581 (2015). [[pdf]](http://arxiv.org/pdf/1511.06581) **(ICLR best paper,great idea)**  :star::star::star::star:

**[48]** Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." arXiv preprint arXiv:1602.01783 (2016). [[pdf]](http://arxiv.org/pdf/1602.01783) **(State-of-the-art method)** :star::star::star::star::star:

**[49]** Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." arXiv preprint arXiv:1509.02971 (2015). [[pdf]](http://arxiv.org/pdf/1509.02971) **(DDPG)** :star::star::star::star:

**[50]** Gu, Shixiang, et al. "**Continuous Deep Q-Learning with Model-based Acceleration**." arXiv preprint arXiv:1603.00748 (2016). [[pdf]](http://arxiv.org/pdf/1603.00748) **(NAF)** :star::star::star::star:

**[51]** Schulman, John, et al. "**Trust region policy optimization**." CoRR, abs/1502.05477 (2015). [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf) **(TRPO)** :star::star::star::star:

**[52]** Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." Nature 529.7587 (2016): 484-489. [[pdf]](http://willamette.edu/~levenick/cs448/goNature.pdf) **(AlphaGo)** :star::star::star::star::star:

## 2.7 Deep Transfer Learning / Lifelong Learning / especially for RL

**[53]** Bengio, Yoshua. "**Deep Learning of Representations for Unsupervised and Transfer Learning**." ICML Unsupervised and Transfer Learning 27 (2012): 17-36. [[pdf]](http://www.jmlr.org/proceedings/papers/v27/bengio12a/bengio12a.pdf) **(A Tutorial)** :star::star::star:

**[54]** Silver, Daniel L., Qiang Yang, and Lianghao Li. "**Lifelong Machine Learning Systems: Beyond Learning Algorithms**." AAAI Spring Symposium: Lifelong Machine Learning. 2013. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7800&rep=rep1&type=pdf) **(A brief discussion about lifelong learning)**  :star::star::star:

**[55]** Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "**Distilling the knowledge in a neural network**." arXiv preprint arXiv:1503.02531 (2015). [[pdf]](http://arxiv.org/pdf/1503.02531) **(Godfather's Work)** :star::star::star::star:

**[56]** Rusu, Andrei A., et al. "**Policy distillation**." arXiv preprint arXiv:1511.06295 (2015). [[pdf]](http://arxiv.org/pdf/1511.06295) **(RL domain)** :star::star::star:

**[57]** Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. "**Actor-mimic: Deep multitask and transfer reinforcement learning**." arXiv preprint arXiv:1511.06342 (2015). [[pdf]](http://arxiv.org/pdf/1511.06342) **(RL domain)** :star::star::star:

**[58]** Rusu, Andrei A., et al. "**Progressive neural networks**." arXiv preprint arXiv:1606.04671 (2016). [[pdf]](https://arxiv.org/pdf/1606.04671) **(Outstanding Work, A novel idea)** :star::star::star::star::star:


## 2.8 One Shot Deep Learning

**[59]** Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) **(No Deep Learning,but worth reading)** :star::star::star::star::star:

**[60]** Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "**Siamese Neural Networks for One-shot Image Recognition**."(2015) [[pdf]](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) :star::star::star:

**[61]** Santoro, Adam, et al. "**One-shot Learning with Memory-Augmented Neural Networks**." arXiv preprint arXiv:1605.06065 (2016). [[pdf]](http://arxiv.org/pdf/1605.06065) **(A basic step to one shot learning)** :star::star::star::star:

**[62]** Vinyals, Oriol, et al. "**Matching Networks for One Shot Learning**." arXiv preprint arXiv:1606.04080 (2016). [[pdf]](https://arxiv.org/pdf/1606.04080) :star::star::star:

**[63]** Hariharan, Bharath, and Ross Girshick. "**Low-shot visual object recognition**." arXiv preprint arXiv:1606.02819 (2016). [[pdf]](http://arxiv.org/pdf/1606.02819) **(A step to large data)** :star::star::star::star:


# 3 Applications

## 3.1 NLP(Natural Language Processing)

**[1]** Antoine Bordes, et al. "**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**." AISTATS(2012) [[pdf]](https://www.hds.utc.fr/~bordesan/dokuwiki/lib/exe/fetch.php?id=en%3Apubli&cache=cache&media=en:bordes12aistats.pdf) :star::star::star::star:

**[2]** Mikolov, et al. "**Distributed representations of words and phrases and their compositionality**." ANIPS(2013): 3111-3119 [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) **(word2vec)** :star::star::star:

**[3]** Sutskever, et al. "**“Sequence to sequence learning with neural networks**." ANIPS(2014) [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :star::star::star:

**[4]** Ankit Kumar, et al. "**“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**." arXiv preprint arXiv:1506.07285(2015) [[pdf]](https://arxiv.org/abs/1506.07285) :star::star::star::star:

**[5]** Yoon Kim, et al. "**Character-Aware Neural Language Models**." NIPS(2015) arXiv preprint arXiv:1508.06615(2015) [[pdf]](https://arxiv.org/abs/1508.06615) :star::star::star::star:

**[6]** Jason Weston, et al. "**Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**." arXiv preprint arXiv:1502.05698(2015) [[pdf]](https://arxiv.org/abs/1502.05698) **(bAbI tasks)** :star::star::star:

**[7]** Karl Moritz Hermann, et al. "**Teaching Machines to Read and Comprehend**." arXiv preprint arXiv:1506.03340(2015) [[pdf]](https://arxiv.org/abs/1506.03340) **(CNN/DailyMail cloze style questions)** :star::star:

**[8]** Alexis Conneau, et al. "**Very Deep Convolutional Networks for Natural Language Processing**." arXiv preprint arXiv:1606.01781(2016) [[pdf]](https://arxiv.org/abs/1606.01781) **(state-of-the-art in text classification)** :star::star::star:

**[9]** Armand Joulin, et al. "**Bag of Tricks for Efficient Text Classification**." arXiv preprint arXiv:1607.01759(2016) [[pdf]](https://arxiv.org/abs/1607.01759) **(slightly worse than state-of-the-art, but a lot faster)** :star::star::star:

## 3.2 Object Detection

**[1]** Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "**Deep neural networks for object detection**." Advances in Neural Information Processing Systems. 2013. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf) :star::star::star:

**[2]** Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) **(RCNN)** :star::star::star::star::star:

**[3]** He, Kaiming, et al. "**Spatial pyramid pooling in deep convolutional networks for visual recognition**." European Conference on Computer Vision. Springer International Publishing, 2014. [[pdf]](http://arxiv.org/pdf/1406.4729) **(SPPNet)** :star::star::star::star:

**[4]** Girshick, Ross. "**Fast r-cnn**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf) :star::star::star::star:

**[5]** Ren, Shaoqing, et al. "**Faster R-CNN: Towards real-time object detection with region proposal networks**." Advances in neural information processing systems. 2015. [[pdf]](https://arxiv.org/pdf/1506.01497.pdf) :star::star::star::star:

**[6]** Redmon, Joseph, et al. "**You only look once: Unified, real-time object detection**." arXiv preprint arXiv:1506.02640 (2015). [[pdf]](http://homes.cs.washington.edu/~ali/papers/YOLO.pdf) **(YOLO,Oustanding Work, really practical)** :star::star::star::star::star:

**[7]** Liu, Wei, et al. "**SSD: Single Shot MultiBox Detector**." arXiv preprint arXiv:1512.02325 (2015). [[pdf]](http://arxiv.org/pdf/1512.02325) :star::star::star:

**[8]** Dai, Jifeng, et al. "**R-FCN: Object Detection via
Region-based Fully Convolutional Networks**." arXiv preprint arXiv:1605.06409 (2016). [[pdf]](https://arxiv.org/abs/1605.06409) :star::star::star::star:

**[9]** He, Gkioxari, et al. "**Mask R-CNN**" arXiv preprint arXiv:1703.06870 (2017). [[pdf]](https://arxiv.org/abs/1703.06870) :star::star::star::star:
## 3.3 Visual Tracking

**[1]** Wang, Naiyan, and Dit-Yan Yeung. "**Learning a deep compact image representation for visual tracking**." Advances in neural information processing systems. 2013. [[pdf]](http://papers.nips.cc/paper/5192-learning-a-deep-compact-image-representation-for-visual-tracking.pdf) **(First Paper to do visual tracking using Deep Learning,DLT Tracker)** :star::star::star:

**[2]** Wang, Naiyan, et al. "**Transferring rich feature hierarchies for robust visual tracking**." arXiv preprint arXiv:1501.04587 (2015). [[pdf]](http://arxiv.org/pdf/1501.04587) **(SO-DLT)** :star::star::star::star:

**[3]** Wang, Lijun, et al. "**Visual tracking with fully convolutional networks**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Visual_Tracking_With_ICCV_2015_paper.pdf) **(FCNT)** :star::star::star::star:

**[4]** Held, David, Sebastian Thrun, and Silvio Savarese. "**Learning to Track at 100 FPS with Deep Regression Networks**." arXiv preprint arXiv:1604.01802 (2016). [[pdf]](http://arxiv.org/pdf/1604.01802) **(GOTURN,Really fast as a deep learning method,but still far behind un-deep-learning methods)** :star::star::star::star:

**[5]** Bertinetto, Luca, et al. "**Fully-Convolutional Siamese Networks for Object Tracking**." arXiv preprint arXiv:1606.09549 (2016). [[pdf]](https://arxiv.org/pdf/1606.09549) **(SiameseFC,New state-of-the-art for real-time object tracking)** :star::star::star::star:

**[6]** Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. "**Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking**." ECCV (2016) [[pdf]](http://www.cvl.isy.liu.se/research/objrec/visualtracking/conttrack/C-COT_ECCV16.pdf) **(C-COT)** :star::star::star::star:

**[7]** Nam, Hyeonseob, Mooyeol Baek, and Bohyung Han. "**Modeling and Propagating CNNs in a Tree Structure for Visual Tracking**." arXiv preprint arXiv:1608.07242 (2016). [[pdf]](https://arxiv.org/pdf/1608.07242) **(VOT2016 Winner,TCNN)** :star::star::star::star:

## 3.4 Image Caption
**[1]** Farhadi,Ali,etal. "**Every picture tells a story: Generating sentences from images**". In Computer VisionECCV 2010. Springer Berlin Heidelberg:15-29, 2010. [[pdf]](https://www.cs.cmu.edu/~afarhadi/papers/sentence.pdf) :star::star::star:

**[2]** Kulkarni, Girish, et al. "**Baby talk: Understanding and generating image descriptions**". In Proceedings of the 24th CVPR, 2011. [[pdf]](http://tamaraberg.com/papers/generation_cvpr11.pdf):star::star::star::star:

**[3]** Vinyals, Oriol, et al. "**Show and tell: A neural image caption generator**". In arXiv preprint arXiv:1411.4555, 2014. [[pdf]](https://arxiv.org/pdf/1411.4555.pdf):star::star::star:

**[4]** Donahue, Jeff, et al. "**Long-term recurrent convolutional networks for visual recognition and description**". In arXiv preprint arXiv:1411.4389 ,2014. [[pdf]](https://arxiv.org/pdf/1411.4389.pdf)

**[5]** Karpathy, Andrej, and Li Fei-Fei. "**Deep visual-semantic alignments for generating image descriptions**". In arXiv preprint arXiv:1412.2306, 2014. [[pdf]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf):star::star::star::star::star:

**[6]** Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. "**Deep fragment embeddings for bidirectional image sentence mapping**". In Advances in neural information processing systems, 2014. [[pdf]](https://arxiv.org/pdf/1406.5679v1.pdf):star::star::star::star:

**[7]** Fang, Hao, et al. "**From captions to visual concepts and back**". In arXiv preprint arXiv:1411.4952, 2014. [[pdf]](https://arxiv.org/pdf/1411.4952v3.pdf):star::star::star::star::star:

**[8]** Chen, Xinlei, and C. Lawrence Zitnick. "**Learning a recurrent visual representation for image caption generation**". In arXiv preprint arXiv:1411.5654, 2014. [[pdf]](https://arxiv.org/pdf/1411.5654v1.pdf):star::star::star::star:

**[9]** Mao, Junhua, et al. "**Deep captioning with multimodal recurrent neural networks (m-rnn)**". In arXiv preprint arXiv:1412.6632, 2014. [[pdf]](https://arxiv.org/pdf/1412.6632v5.pdf):star::star::star:

**[10]** Xu, Kelvin, et al. "**Show, attend and tell: Neural image caption generation with visual attention**". In arXiv preprint arXiv:1502.03044, 2015. [[pdf]](https://arxiv.org/pdf/1502.03044v3.pdf):star::star::star::star::star:

## 3.5 Machine Translation

> Some milestone papers are listed in RNN / Seq-to-Seq topic.

**[1]** Luong, Minh-Thang, et al. "**Addressing the rare word problem in neural machine translation**." arXiv preprint arXiv:1410.8206 (2014). [[pdf]](http://arxiv.org/pdf/1410.8206) :star::star::star::star:


**[2]** Sennrich, et al. "**Neural Machine Translation of Rare Words with Subword Units**". In arXiv preprint arXiv:1508.07909, 2015. [[pdf]](https://arxiv.org/pdf/1508.07909.pdf):star::star::star:

**[3]** Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "**Effective approaches to attention-based neural machine translation**." arXiv preprint arXiv:1508.04025 (2015). [[pdf]](http://arxiv.org/pdf/1508.04025) :star::star::star::star:

**[4]** Chung, et al. "**A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation**". In arXiv preprint arXiv:1603.06147, 2016. [[pdf]](https://arxiv.org/pdf/1603.06147.pdf):star::star:

**[5]** Lee, et al. "**Fully Character-Level Neural Machine Translation without Explicit Segmentation**". In arXiv preprint arXiv:1610.03017, 2016. [[pdf]](https://arxiv.org/pdf/1610.03017.pdf):star::star::star::star::star:

**[6]** Wu, Schuster, Chen, Le, et al. "**Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation**". In arXiv preprint arXiv:1609.08144v2, 2016. [[pdf]](https://arxiv.org/pdf/1609.08144v2.pdf) **(Milestone)** :star::star::star::star:

## 3.6 Robotics

**[1]** Koutník, Jan, et al. "**Evolving large-scale neural networks for vision-based reinforcement learning**." Proceedings of the 15th annual conference on Genetic and evolutionary computation. ACM, 2013. [[pdf]](http://repository.supsi.ch/4550/1/koutnik2013gecco.pdf) :star::star::star:

**[2]** Levine, Sergey, et al. "**End-to-end training of deep visuomotor policies**." Journal of Machine Learning Research 17.39 (2016): 1-40. [[pdf]](http://www.jmlr.org/papers/volume17/15-522/15-522.pdf) :star::star::star::star::star:

**[3]** Pinto, Lerrel, and Abhinav Gupta. "**Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours**." arXiv preprint arXiv:1509.06825 (2015). [[pdf]](http://arxiv.org/pdf/1509.06825) :star::star::star:

**[4]** Levine, Sergey, et al. "**Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection**." arXiv preprint arXiv:1603.02199 (2016). [[pdf]](http://arxiv.org/pdf/1603.02199) :star::star::star::star:

**[5]** Zhu, Yuke, et al. "**Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning**." arXiv preprint arXiv:1609.05143 (2016). [[pdf]](https://arxiv.org/pdf/1609.05143) :star::star::star::star:

**[6]** Yahya, Ali, et al. "**Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search**." arXiv preprint arXiv:1610.00673 (2016). [[pdf]](https://arxiv.org/pdf/1610.00673) :star::star::star::star:

**[7]** Gu, Shixiang, et al. "**Deep Reinforcement Learning for Robotic Manipulation**." arXiv preprint arXiv:1610.00633 (2016). [[pdf]](https://arxiv.org/pdf/1610.00633) :star::star::star::star:

**[8]** A Rusu, M Vecerik, Thomas Rothörl, N Heess, R Pascanu, R Hadsell."**Sim-to-Real Robot Learning from Pixels with Progressive Nets**." arXiv preprint arXiv:1610.04286 (2016). [[pdf]](https://arxiv.org/pdf/1610.04286.pdf) :star::star::star::star:

**[9]** Mirowski, Piotr, et al. "**Learning to navigate in complex environments**." arXiv preprint arXiv:1611.03673 (2016). [[pdf]](https://arxiv.org/pdf/1611.03673) :star::star::star::star:

## 3.7 Art

**[1]** Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). "**Inceptionism: Going Deeper into Neural Networks**". Google Research. [[html]](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) **(Deep Dream)**
:star::star::star::star:

**[2]** Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "**A neural algorithm of artistic style**." arXiv preprint arXiv:1508.06576 (2015). [[pdf]](http://arxiv.org/pdf/1508.06576) **(Outstanding Work, most successful method currently)** :star::star::star::star::star:

**[3]** Zhu, Jun-Yan, et al. "**Generative Visual Manipulation on the Natural Image Manifold**." European Conference on Computer Vision. Springer International Publishing, 2016. [[pdf]](https://arxiv.org/pdf/1609.03552) **(iGAN)** :star::star::star::star:

**[4]** Champandard, Alex J. "**Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks**." arXiv preprint arXiv:1603.01768 (2016). [[pdf]](http://arxiv.org/pdf/1603.01768) **(Neural Doodle)** :star::star::star::star:

**[5]** Zhang, Richard, Phillip Isola, and Alexei A. Efros. "**Colorful Image Colorization**." arXiv preprint arXiv:1603.08511 (2016). [[pdf]](http://arxiv.org/pdf/1603.08511) :star::star::star::star:

**[6]** Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "**Perceptual losses for real-time style transfer and super-resolution**." arXiv preprint arXiv:1603.08155 (2016). [[pdf]](https://arxiv.org/pdf/1603.08155.pdf) :star::star::star::star:

**[7]** Vincent Dumoulin, Jonathon Shlens and Manjunath Kudlur. "**A learned representation for artistic style**." arXiv preprint arXiv:1610.07629 (2016). [[pdf]](https://arxiv.org/pdf/1610.07629v1.pdf) :star::star::star::star:

**[8]** Gatys, Leon and Ecker, et al."**Controlling Perceptual Factors in Neural Style Transfer**." arXiv preprint arXiv:1611.07865 (2016). [[pdf]](https://arxiv.org/pdf/1611.07865.pdf) **(control style transfer over spatial location,colour information and across spatial scale)**:star::star::star::star:

**[9]** Ulyanov, Dmitry and Lebedev, Vadim, et al. "**Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**." arXiv preprint arXiv:1603.03417(2016). [[pdf]](http://arxiv.org/abs/1603.03417) **(texture generation and style transfer)** :star::star::star::star:


## 3.8 Object Segmentation

**[1]** J. Long, E. Shelhamer, and T. Darrell, “**Fully convolutional networks for semantic segmentation**.” in CVPR, 2015. [[pdf]](https://arxiv.org/pdf/1411.4038v2.pdf) :star::star::star::star::star:

**[2]** L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. "**Semantic image segmentation with deep convolutional nets and fully connected crfs**." In ICLR, 2015. [[pdf]](https://arxiv.org/pdf/1606.00915v1.pdf) :star::star::star::star::star:

**[3]** Pinheiro, P.O., Collobert, R., Dollar, P. "**Learning to segment object candidates.**" In: NIPS. 2015. [[pdf]](https://arxiv.org/pdf/1506.06204v2.pdf) :star::star::star::star:

**[4]** Dai, J., He, K., Sun, J. "**Instance-aware semantic segmentation via multi-task network cascades**." in CVPR. 2016 [[pdf]](https://arxiv.org/pdf/1512.04412v1.pdf) :star::star::star:

**[5]** Dai, J., He, K., Sun, J. "**Instance-sensitive Fully Convolutional Networks**." arXiv preprint arXiv:1603.08678 (2016). [[pdf]](https://arxiv.org/pdf/1603.08678v1.pdf) :star::star::star:






    如果你现在还是个深度学习的新手，那么你问的第一个问题可能是「我应该从哪篇文章开始读呢？」在 Github 上，songrotek 准备了一套深度学习阅读清单，而且这份清单在随时更新。至于文中提到的 PDF，读者们可点击阅读原文下载机器之心打包的论文，或点开下面的项目地址下载自己喜欢的学习材料。


项目地址：https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap


这份清单依照下述 4 条原则建立：


    从整体轮廓到细节

    从过去到当代

    从一般到具体领域

    聚焦当下最先进技术


你会发现很多非常新但很值得一读的论文。这份清单我会持续更新。


1、深度学习的历史与基础知识

1.0 书籍


[0] Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. 深度学习（Deep learning）, An MIT Press book. (2015). （这是深度学习领域的圣经，你可以在读此书的同时阅读下面的论文）。


1.1 调查类：


[1] LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. 深度学习 (Deep learning), Nature 521.7553 (2015): 436-444. (深度学习三位大牛对各种学习模型的评价)


1.2 深度信念网络（DBN）（深度学习前夜的里程碑）


[2] Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. 一个关于深度信念网络的快速学习算法（A fast learning algorithm for deep belief nets）, (深度学习的前夜)


[3] Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. 使用神经网络降低数据的维度（Reducing the dimensionality of data with neural networks）,  (里程碑式的论文，展示了深度学习的可靠性)


1.3 ImageNet 的演化（深度学习从这里开始）


[4] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. 使用深度卷积神经网络进行 ImageNet 分类任务（Imagenet classification with deep convolutional neural networks）(AlexNet, 深度学习的突破)


[5] Simonyan, Karen, and Andrew Zisserman. 针对大尺度图像识别工作的的超深卷积网络（Very deep convolutional networks for large-scale image recognition） (VGGNet, 神经网络开始变得非常深！)


[6] Szegedy, Christian, et al. 更深的卷积（Going deeper with convolutions）(GoogLeNet)


[7] He, Kaiming, et al. 图像识别的深度残差学习（Deep residual learning for image recognition）(ResNet，超级超级深的深度网络！CVPR--IEEE 国际计算机视觉与模式识别会议-- 最佳论文)


1.4 语音识别的演化


[8] Hinton, Geoffrey, et al. 语音识别中深度神经网络的声学建模（Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups）（语音识别中的突破)


[9] Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. 用深度循环神经网络进行语音识别（Speech recognition with deep recurrent neural networks）(RNN)


[10] Graves, Alex, and Navdeep Jaitly. 面向端到端语音识别的循环神经网络（Towards End-To-End Speech Recognition with Recurrent Neural Networks）


[11] Sak, Haşim, et al. 语音识别中快且精准的循环神经网络声学模型（Fast and accurate recurrent neural network acoustic models for speech recognition）(谷歌语音识别系统)


[12] Amodei, Dario, et al. Deep speech 2:英语和汉语的端到端语音识别（Deep speech 2: End-to-end speech recognition in english and mandarin）(百度语音识别系统)


[13] W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig，在对话语音识别中实现人类平等（Achieving Human Parity in Conversational Speech Recognition） (最先进的语音识别技术，微软)


当你读完了上面给出的论文，你会对深度学习历史有一个基本的了解，深度学习建模的基本架构（包括了 CNN，RNN，LSTM）以及深度学习如何可以被应用于图像和语音识别问题。下面的论文会让你对深度学习方法，不同应用领域中的深度学习技术和其局限有深度认识。我建议你可以基于自己的兴趣和研究方向选择下面这些论文。


2 深度学习方法


2.1 模型


[14] Hinton, Geoffrey E., et al. 通过避免特征检测器的共适应来改善神经网络（Improving neural networks by preventing co-adaptation of feature detectors）(Dropout)


[15] Srivastava, Nitish, et al. Dropout：一种避免神经网络过度拟合的简单方法（Dropout: a simple way to prevent neural networks from overfitting）


[16] Ioffe, Sergey, and Christian Szegedy. Batch normalization:通过减少内部协变量加速深度网络训练（Batch normalization: Accelerating deep network training by reducing internal covariate shift）(2015 年一篇杰出论文)


[17] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.层归一化（Layer normalization）(批归一化的升级版)


[18] Courbariaux, Matthieu, et al. 二值神经网络：训练神经网络的权重和激活约束到正 1 或者负 1（Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1）(新模型，快)


[19] Jaderberg, Max, et al. 使用合成梯度的解耦神经接口（Decoupled neural interfaces using synthetic gradients）(训练方法的发明，令人惊叹的文章)


[20] Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. Net2net：通过知识迁移加速学习（Net2net: Accelerating learning via knowledge transfer） (修改之前的训练网络以减少训练)


[21] Wei, Tao, et al. 网络形态（Network Morphism）(修改之前的训练网络以减少训练 epoch)


2.2 优化


[22] Sutskever, Ilya, et al. 有关深度学习中初始化与动量因子的研究（On the importance of initialization and momentum in deep learning） (动量因子优化器)


[23] Kingma, Diederik, and Jimmy Ba. Adam：随机优化的一种方法（Adam: A method for stochastic optimization）(可能是现在用的最多的一种方法)


[24] Andrychowicz, Marcin, et al. 通过梯度下降学习梯度下降（Learning to learn by gradient descent by gradient descent） (神经优化器，令人称奇的工作)


[25] Han, Song, Huizi Mao, and William J. Dally. 深度压缩：通过剪枝、量子化训练和霍夫曼代码压缩深度神经网络（Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding） (ICLR 最佳论文，来自 DeePhi 科技初创公司，加速 NN 运行的新方向)


[26] Iandola, Forrest N., et al. SqueezeNet：带有 50x 更少参数和小于 1MB 模型大小的 AlexNet-层级精确度（SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size.） (优化 NN 的另一个新方向，来自 DeePhi 科技初创公司)


2.3 无监督学习／深度生成模型


[27] Le, Quoc V. 通过大规模无监督学习构建高级特征（Building high-level features using large scale unsupervised learning.） (里程碑，吴恩达，谷歌大脑，猫)


[28] Kingma, Diederik P., and Max Welling. 自动编码变异贝叶斯（Auto-encoding variational bayes.） (VAE) 


[29] Goodfellow, Ian, et al. 生成对抗网络（Generative adversarial nets.）(GAN, 超酷的想法)


[30] Radford, Alec, Luke Metz, and Soumith Chintala. 带有深度卷曲生成对抗网络的无监督特征学习（Unsupervised representation learning with deep convolutional generative adversarial networks.）(DCGAN)


[31] Gregor, Karol, et al. DRAW：一个用于图像生成的循环神经网络（DRAW: A recurrent neural network for image generation.） (值得注意的 VAE，杰出的工作)


[32] Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. 像素循环神经网络（Pixel recurrent neural networks.）(像素 RNN)


[33] Oord, Aaron van den, et al. 使用像素 CNN 解码器有条件地生成图像（Conditional image generation with PixelCNN decoders.） (像素 CNN)


2.4 RNN／序列到序列模型


[34] Graves, Alex. 带有循环神经网络的生成序列（Generating sequences with recurrent neural networks.）(LSTM, 非常好的生成结果，展示了 RNN 的力量)


[35] Cho, Kyunghyun, et al. 使用 RNN 编码器-解码器学习词组表征用于统计机器翻译（Learning phrase representations using RNN encoder-decoder for statistical machine translation.） (第一个序列到序列论文)


[36] Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. 运用神经网路的序列到序列学习（Sequence to sequence learning with neural networks.」）(杰出的工作)


[37] Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. 通过共同学习来匹配和翻译神经机器翻译（Neural Machine Translation by Jointly Learning to Align and Translate.）


[38] Vinyals, Oriol, and Quoc Le. 一个神经对话模型（A neural conversational model.）(聊天机器人上的序列到序列)


2.5 神经图灵机


[39] Graves, Alex, Greg Wayne, and Ivo Danihelka. 神经图灵机器（Neural turing machines.）arXiv preprint arXiv:1410.5401 (2014). (未来计算机的基本原型）


[40] Zaremba, Wojciech, and Ilya Sutskever. 强化学习神经图灵机（Reinforcement learning neural Turing machines.）


[41] Weston, Jason, Sumit Chopra, and Antoine Bordes. 记忆网络（Memory networks.）


[42] Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. 端到端记忆网络（End-to-end memory networks.）


[43] Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. 指示器网络（Pointer networks.）


[44] Graves, Alex, et al. 使用带有动力外部内存的神经网络的混合计算（Hybrid computing using a neural network with dynamic external memory.）(里程碑，结合上述论文的思想)


2.6 深度强化学习


[45] Mnih, Volodymyr, et al. 使用深度强化学习玩 atari 游戏（Playing atari with deep reinforcement learning.） (第一篇以深度强化学习命名的论文)


[46] Mnih, Volodymyr, et al. 通过深度强化学习达到人类水准的控制（Human-level control through deep reinforcement learning.） (里程碑)


[47] Wang, Ziyu, Nando de Freitas, and Marc Lanctot. 用于深度强化学习的决斗网络架构（Dueling network architectures for deep reinforcement learning.） (ICLR 最佳论文，伟大的想法 )


[48] Mnih, Volodymyr, et al. 用于深度强化学习的异步方法（Asynchronous methods for deep reinforcement learning.） (当前最先进的方法)


[49] Lillicrap, Timothy P., et al. 运用深度强化学习进行持续控制（Continuous control with deep reinforcement learning.） (DDPG) 


[50] Gu, Shixiang, et al. 带有模型加速的持续深层 Q-学习（Continuous Deep Q-Learning with Model-based Acceleration.）


[51] Schulman, John, et al. 信赖域策略优化（Trust region policy optimization.） (TRPO)


[52] Silver, David, et al. 使用深度神经网络和树搜索掌握围棋游戏（Mastering the game of Go with deep neural networks and tree search.） (阿尔法狗)


2.7 深度迁移学习／终身学习／尤其对于 RL


[53] Bengio, Yoshua. 表征无监督和迁移学习的深度学习（Deep Learning of Representations for Unsupervised and Transfer Learning.） (一个教程)


[54] Silver, Daniel L., Qiang Yang, and Lianghao Li. 终身机器学习系统：超越学习算法（Lifelong Machine Learning Systems: Beyond Learning Algorithms.） (一个关于终生学习的简要讨论)


[55] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 提取神经网络中的知识（Distilling the knowledge in a neural network.） (教父的工作) 


[56] Rusu, Andrei A., et al. 策略提取（Policy distillation.） (RL 领域)


[57] Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. 演员模仿：深度多任务和迁移强化学习（Actor-mimic: Deep multitask and transfer reinforcement learning.） (RL 领域)


[58] Rusu, Andrei A., et al. 渐进神经网络（Progressive neural networks.）(杰出的工作，一项全新的工作)


2.8 一次性深度学习


[59] Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. 通过概率程序归纳达到人类水准的概念学习（Human-level concept learning through probabilistic program induction.）(不是深度学习，但是值得阅读)


[60] Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. 用于一次图像识别的孪生神经网络（Siamese Neural Networks for One-shot Image Recognition.）


[61] Santoro, Adam, et al. 用记忆增强神经网络进行一次性学习（One-shot Learning with Memory-Augmented Neural Networks ） (一个一次性学习的基本步骤)


[62] Vinyals, Oriol, et al. 用于一次性学习的匹配网络（Matching Networks for One Shot Learning.）


[63] Hariharan, Bharath, and Ross Girshick. 少量视觉物体识别（Low-shot visual object recognition.）(走向大数据的一步)


3 应用


3.1 NLP（自然语言处理）


[1] Antoine Bordes, et al. 开放文本语义分析的词和意义表征的联合学习（Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing.）


[2] Mikolov, et al. 词和短语及其组合性的分布式表征（Distributed representations of words and phrases and their compositionality.） (word2vec)


[3] Sutskever, et al. 运用神经网络的序列到序列学习（Sequence to sequence learning with neural networks.）


[4] Ankit Kumar, et al. 问我一切：动态记忆网络用于自然语言处理（Ask Me Anything: Dynamic Memory Networks for Natural Language Processing.）


[5] Yoon Kim, et al. 角色意识的神经语言模型（Character-Aware Neural Language Models.）


[6] Jason Weston, et al. 走向人工智能-完成问题回答：一组前提玩具任务（Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks.） (bAbI 任务)


[7] Karl Moritz Hermann, et al. 教机器阅读和理解（Teaching Machines to Read and Comprehend.）(CNN/每日邮件完形风格问题)


[8] Alexis Conneau, et al. 非常深度卷曲网络用于自然语言处理（Very Deep Convolutional Networks for Natural Language Processing.） (在文本分类中当前最好的)


[9] Armand Joulin, et al. 诡计包用于有效文本分类（Bag of Tricks for Efficient Text Classification.）(比最好的差一点，但快很多)


3.2 目标检测


[1] Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. 深度神经网路用于目标检测（Deep neural networks for object detection.）


[2] Girshick, Ross, et al. 富特征层级用于精确目标检测和语义分割（Rich feature hierarchies for accurate object detection and semantic segmentation.）(RCNN)


[3] He, Kaiming, et al. 深度卷曲网络的空间金字塔池用于视觉识别（Spatial pyramid pooling in deep convolutional networks for visual recognition.） (SPPNet)


[4] Girshick, Ross. 快速的循环卷曲神经网络（Fast r-cnn.）


[5] Ren, Shaoqing, et al. 更快的循环卷曲神经网络：通过区域建议网络趋向实时目标检测（Faster R-CNN: Towards real-time object detection with region proposal networks.）


[6] Redmon, Joseph, et al. 你只看到一次：统一实时的目标检测（You only look once: Unified, real-time object detection.） (YOLO, 杰出的工作，真的很实用)


[7] Liu, Wei, et al. SSD：一次性多盒探测器（SSD: Single Shot MultiBox Detector.）


3.3 视觉跟踪


[1] Wang, Naiyan, and Dit-Yan Yeung. 学习视觉跟踪用的一种深度压缩图象表示（Learning a deep compact image representation for visual tracking.） (第一篇使用深度学习进行视觉跟踪的论文，DLT 跟踪器)


[2] Wang, Naiyan, et al. 为稳定的视觉跟踪传输丰富特征层次（Transferring rich feature hierarchies for robust visual tracking.）(SO-DLT)


[3] Wang, Lijun, et al. 用全卷积网络进行视觉跟踪（Visual tracking with fully convolutional networks.） (FCNT)


[4] Held, David, Sebastian Thrun, and Silvio Savarese. 用深度回归网络以 100FPS 速度跟踪（Learning to Track at 100 FPS with Deep Regression Networks.） (GOTURN, 作为一个深度神经网络，其速度非常快，但是相较于非深度学习方法还是慢了很多)


[5] Bertinetto, Luca, et al. 对象跟踪的全卷积 Siamese 网络（Fully-Convolutional Siamese Networks for Object Tracking.） (SiameseFC, 实时对象追踪的最先进技术)


[6] Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. 超越相关滤波器：学习连续卷积算子的视觉追踪（Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking.）(C-COT)


[7] Nam, Hyeonseob, Mooyeol Baek, and Bohyung Han. 在视觉跟踪的树结构中传递卷积神经网络与建模（Modeling and Propagating CNNs in a Tree Structure for Visual Tracking.）(VOT2016 Winner,TCNN)


3.4 图像说明


[1] Farhadi,Ali,etal. 每幅图都讲述了一个故事：从图像中生成句子（Every picture tells a story: Generating sentences from images.）


[2] Kulkarni, Girish, et al. 儿语：理解并生成图像的描述（talk: Understanding and generating image descriptions.）


[3] Vinyals, Oriol, et al. 展示与表达：一个神经图像字幕生成器（Show and tell: A neural image caption generator）


[4] Donahue, Jeff, et al. 视觉认知和描述的长期递归卷积网络（Long-term recurrent convolutional networks for visual recognition and description）


[5] Karpathy, Andrej, and Li Fei-Fei. 产生图像描述的深层视觉语义对齐（Deep visual-semantic alignments for generating image descriptions）


[6] Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. 双向图像句映射的深片段嵌入（Deep fragment embeddings for bidirectional image sentence mapping）


[7] Fang, Hao, et al. 从字幕到视觉概念，从视觉概念到字幕（From captions to visual concepts and back）


[8] Chen, Xinlei, and C. Lawrence Zitnick. 图像字幕生成的递归视觉表征学习「Learning a recurrent visual representation for image caption generation


[9] Mao, Junhua, et al. 使用多模型递归神经网络（m-rnn）的深度字幕生成（Deep captioning with multimodal recurrent neural networks (m-rnn).）


[10] Xu, Kelvin, et al. 展示、参与与表达：视觉注意的神经图像字幕生成（Show, attend and tell: Neural image caption generation with visual attention.）


3.5 机器翻译


一些里程碑式的论文在 RNN \序列到序列的主题分类下被列举。


[1] Luong, Minh-Thang, et al. 神经机器翻译中生僻词问题的处理（Addressing the rare word problem in neural machine translation.）


[2] Sennrich, et al. 带有子词单元的生僻字神经机器翻译（Neural Machine Translation of Rare Words with Subword Units）


[3] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. 基于注意力的神经机器翻译的有效途径（Effective approaches to attention-based neural machine translation.）


[4] Chung, et al. 一个机器翻译无显式分割的字符级解码器（A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation）


[5] Lee, et al. 无显式分割的全字符级神经机器翻译（Fully Character-Level Neural Machine Translation without Explicit Segmentation）


[6] Wu, Schuster, Chen, Le, et al. 谷歌的神经机器翻译系统：弥合人与机器翻译的鸿沟（Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation）


3.6 机器人


[1] Koutník, Jan, et al. 发展用于视觉强化学习的大规模神经网络（Evolving large-scale neural networks for vision-based reinforcement learning.）


[2] Levine, Sergey, et al. 深度视觉眼肌运动策略的端到端训练（End-to-end training of deep visuomotor policies.）


[3] Pinto, Lerrel, and Abhinav Gupta. 超大尺度自我监督：从 5 万次尝试和 700 机器人小时中学习抓取（Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours.）


[4] Levine, Sergey, et al. 学习手眼协作用于机器人掌握深度学习和大数据搜集（Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection.）


[5] Zhu, Yuke, et al. 使用深度强化学习视觉导航目标驱动的室内场景（Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning.）


[6] Yahya, Ali, et al. 使用分布式异步引导策略搜索进行集体机器人增强学习（Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search.）


[7] Gu, Shixiang, et al. 深度强化学习用于机器操控（Deep Reinforcement Learning for Robotic Manipulation.）


[8] A Rusu, M Vecerik, Thomas Rothörl, N Heess, R Pascanu, R Hadsell. 模拟实机机器人使用过程网从像素中学习（Sim-to-Real Robot Learning from Pixels with Progressive Nets.）


[9] Mirowski, Piotr, et al. 学习在复杂环境中导航（Learning to navigate in complex environments.）


3.7 艺术


[1] Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). 初始主义：神经网络的更深层（Inceptionism: Going Deeper into Neural Networks）(谷歌 Deep Dream) 


[2] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. 一个艺术风格的神经算法（A neural algorithm of artistic style.） (杰出的工作，目前最成功的算法) 


[3] Zhu, Jun-Yan, et al. 自然图像流形上的生成视觉操纵（Generative Visual Manipulation on the Natural Image Manifold.）


[4] Champandard, Alex J. Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks. (神经涂鸦) 


[5] Zhang, Richard, Phillip Isola, and Alexei A. Efros. 多彩的图像彩色化（Colorful Image Colorization.）


[6] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. 实时风格迁移和超分辨率的感知损失（Perceptual losses for real-time style transfer and super-resolution.）


[7] Vincent Dumoulin, Jonathon Shlens and Manjunath Kudlur. 一个艺术风格的学习表征（A learned representation for artistic style.）


[8] Gatys, Leon and Ecker, et al. 神经风格迁移中的控制感知因子（Controlling Perceptual Factors in Neural Style Transfer.） (控制空间定位、色彩信息和全空间尺度方面的风格迁移)


[9] Ulyanov, Dmitry and Lebedev, Vadim, et al. 纹理网络：纹理和风格化图像的前馈合成（Texture Networks: Feed-forward Synthesis of Textures and Stylized Images.） (纹理生成和风格迁移)


3.8 对象分割


[1] J. Long, E. Shelhamer, and T. Darrell, 用于语义分割的全卷积网络（Fully convolutional networks for semantic segmentation）


[2] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. 具有深度卷积网络和全连接的条件随机场的语义图像分割（Semantic image segmentation with deep convolutional nets and fully connected crfs）


[3] Pinheiro, P.O., Collobert, R., Dollar, P. 学习如何分割候选对象（Learning to segment object candidates）


[4] Dai, J., He, K., Sun, J. 基于多任务网络级联的实例感知语义分割（Instance-aware semantic segmentation via multi-task network cascades）


[5] Dai, J., He, K., Sun, J. 实例敏感的全卷积网络（Instance-sensitive Fully Convolutional Networks）
