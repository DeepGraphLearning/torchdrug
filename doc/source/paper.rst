Papers Implemented
==================

.. include:: bibliography.rst

Graph Representation Learning
-----------------------------

Graph Neural Networks
^^^^^^^^^^^^^^^^^^^^^

1. `Convolutional Networks on Graphs for Learning Molecular Fingerprints <NFP_>`_

   David Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Gómez-Bombarelli, Timothy Hirzel, Alán Aspuru-Guzik, Ryan P. Adams. NIPS 2015.

   :class:`NeuralFingerprintConv <torchdrug.layers.NeuralFingerprintConv>`,
   :class:`NeuralFingerprint <torchdrug.models.NeuralFingerprint>`

2. `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering <ChebNet_>`_

   Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst. NIPS 2016.

   :class:`ChebyshevConv <torchdrug.layers.ChebyshevConv>`,
   :class:`ChebyshevConvolutionalNetwork <torchdrug.models.ChebyshevConvolutionalNetwork>`

3. `Semi-Supervised Classification with Graph Convolutional Networks <GCN_>`_

   Thomas N. Kipf, Max Welling. ICLR 2017.

   :class:`GraphConv <torchdrug.layers.GraphConv>`,
   :class:`GraphConvolutionalNetwork <torchdrug.models.GraphConvolutionalNetwork>`

4. `Neural Message Passing for Quantum Chemistry <ENN-S2S_>`_

   Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl. ICML 2017.

   :class:`MessagePassing <torchdrug.layers.MessagePassing>`,
   :class:`MessagePassingNeuralNetwork <torchdrug.models.MessagePassingNeuralNetwork>`

5. `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions <SchNet_>`_

   Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko,
   Klaus-Robert Müller. NeurIPS 2017.

   :class:`ContinuousFilterConv <torchdrug.layers.ContinuousFilterConv>`,
   :class:`SchNet <torchdrug.models.SchNet>`
   
6. `Graph Attention Networks <GAT_>`_

   Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio. ICLR 2018.

   :class:`GraphAttentionConv <torchdrug.layers.GraphAttentionConv>`,
   :class:`GraphAttentionNetwork <torchdrug.models.GraphAttentionNetwork>`

7. `Modeling Relational Data with Graph Convolutional Networks <RGCN_>`_

   Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling. ESWC 2018.

   :class:`RelationalGraphConv <torchdrug.layers.RelationalGraphConv>`,
   :class:`RelationalGraphConvolutionalNetwork <torchdrug.models.RelationalGraphConvolutionalNetwork>`

8. `How Powerful Are Graph Neural Nerworks? <GIN_>`_

   Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka. ICLR 2019.

   :class:`GraphIsomorphismConv <torchdrug.layers.GraphIsomorphismConv>`,
   :class:`GraphIsomorphismNetwork <torchdrug.models.GraphIsomorphismNetwork>`

Differentiable Graph Pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. `Hierarchical Graph Representation Learning with Differentiable Pooling <DiffPool_>`_

   Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec. NeurIPS 2018.

   :class:`DiffPool <torchdrug.layers.DiffPool>`

2. `Spectral Clustering with Graph Neural Networks for Graph Pooling <MinCutPool_>`_

   Filippo Maria Bianchi, Daniele Grattarola, Cesare Alippi. ICML 2020.

   :class:`MinCutPool <torchdrug.layers.MinCutPool>`

Readout Layers
^^^^^^^^^^^^^^

1. `Order Matters: Sequence to sequence for sets <Set2Set_>`_

   Oriol Vinyals, Samy Bengio, Manjunath Kudlur

   :class:`Set2Set <torchdrug.layers.Set2Set>`

Normalization Layers
^^^^^^^^^^^^^^^^^^^^

1. `PairNorm: Tackling Oversmoothing in GNNs <PairNorm_>`_

   Lingxiao Zhao, Leman Akoglu. ICLR 2020.

   :class:`PairNorm <torchdrug.layers.PairNorm>`


Drug Discovery
--------------

Pretrain Molecular Representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. `InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization <InfoGraph_>`_

   Fan-Yun Sun, Jordan Hoffman, Vikas Verma, Jian Tang. ICLR 2020.

   :class:`InfoGraph <torchdrug.models.InfoGraph>`

2. `Strategies for Pre-training Graph Neural Networks <AttrMasking_>`_

   Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec. ICLR 2020.

   :class:`EdgePrediction <torchdrug.tasks.EdgePrediction>`,
   :class:`AttributeMasking <torchdrug.tasks.AttributeMasking>`,
   :class:`ContextPrediction <torchdrug.tasks.ContextPrediction>`

De Novo Molecule Design
^^^^^^^^^^^^^^^^^^^^^^^

1. `Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation. <GCPN_>`_
  
   Jiaxuan You, Bowen Liu, Rex Ying, Vijay Pande, Jure Leskovec. NeurIPS 2018.

   :class:`GCPNGeneration <torchdrug.tasks.GCPNGeneration>`

2. `GraphAF: A Flow-based Autoregressive Model for Molecular Graph Generation. <GraphAF_>`_
  
   Chence Shi, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, Jian Tang. ICLR 2020.

   :class:`GraphAutoregressiveFlow <torchdrug.models.GraphAutoregressiveFlow>`,
   :class:`AutoregressiveGeneration <torchdrug.tasks.AutoregressiveGeneration>`

Retrosynthesis
^^^^^^^^^^^^^^

1. `A Graph to Graphs Framework for Retrosynthesis Prediction. <G2Gs_>`_
 
   Chence Shi, Minkai Xu, Hongyu Guo, Ming Zhang, Jian Tang. ICML 2020.

   :class:`CenterIdentification <torchdrug.tasks.CenterIdentification>`,
   :class:`SynthonCompletion <torchdrug.tasks.SynthonCompletion>`,
   :class:`Retrosynthesis <torchdrug.tasks.Retrosynthesis>`

Protein Representation Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. `Evaluating Protein Transfer Learning with TAPE <TAPE_>`_

   Roshan Rao, Nicholas Bhattacharya, Neil Thomas, Yan Duan, Xi Chen, John Canny, Pieter Abbeel, Yun S Song. NeurIPS 2019.

   :class:`SinusoidalPositionEmbedding <torchdrug.layers.SinusoidalPositionEmbedding>`
   :class:`SelfAttentionBlock <torchdrug.layers.SelfAttentionBlock>`
   :class:`ProteinResNetBlock <torchdrug.layers.ProteinResNetBlock>`
   :class:`ProteinBERTBlock <torchdrug.layers.ProteinBERTBlock>`
   :class:`ProteinResNet <torchdrug.models.ProteinResNet>`
   :class:`ProteinLSTM <torchdrug.models.ProteinLSTM>`
   :class:`ProteinBERT <torchdrug.models.ProteinBERT>`

2. `Is Transfer Learning Necessary for Protein Landscape Prediction? <ProteinCNN_>`_

   Amir Shanehsazzadeh, David Belanger, David Dohan. arXiv 2020.

   :class:`ProteinCNN <torchdrug.models.ProteinCNN>`

3. `Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences <ESM_>`_

   Alexander Rives,  Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, Rob Fergus. PNAS 2021.

   :class:`EvolutionaryScaleModeling <torchdrug.models.EvolutionaryScaleModeling>`

4. `Protein Representation Learning by Geometric Structure Pretraining <GearNet_>`_

   Zuobai Zhang, Minghao Xu, Arian Jamasb, Vijil Chenthamarakshan, Aurélie Lozano, Payel Das, Jian Tang. arXiv 2022.

   :class:`GeometricRelationalGraphConv <torchdrug.layers.GeometricRelationalGraphConv>`
   :class:`GeometryAwareRelationalGraphNeuralNetwork <torchdrug.models.GeometryAwareRelationalGraphNeuralNetwork>`
   :mod:`torchdrug.layers.geometry`

Knowledge Graph Reasoning
^^^^^^^^^^^^^^^^^^^^^^^^^

1. `Translating Embeddings for Modeling Multi-relational Data <TransE_>`_

   Antoine Bordes, Nicolas Usunier, Alberto García-Durán. NIPS 2013.

   :func:`transe_score <torchdrug.layers.functional.transe_score>`,
   :class:`TransE <torchdrug.models.DistMult>`

2. `Embedding Entities and Relations for Learning and Inference in Knowledge Bases <DistMult_>`_

   Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng. ICLR 2015.

   :func:`distmult_score <torchdrug.layers.functional.distmult_score>`,
   :class:`DistMult <torchdrug.models.DistMult>`

3. `Complex Embeddings for Simple Link Prediction <ComplEx_>`_

   Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, Guillaume Bouchard. ICML 2016.

   :func:`complex_score <torchdrug.layers.functional.complex_score>`,
   :class:`ComplEx <torchdrug.models.DistMult>`

4. `Differentiable Learning of Logical Rules for Knowledge Base Reasoning <NeuralLP_>`_

   Fan Yang, Zhilin Yang, William W. Cohen. NIPS 2017.

   :class:`NeuralLogicProgramming <torchdrug.models.NeuralLogicProgramming>`
   
5. `SimplE Embedding for Link Prediction in Knowledge Graphs <SimplE_>`_

   Seyed Mehran Kazemi, David Poole. NeurIPS 2018.

   :func:`simple_score <torchdrug.layers.functional.simple_score>`,
   :class:`SimplE <torchdrug.models.SimplE>`

6. `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space <RotatE_>`_

   Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, Jian Tang. ICLR 2019.

   :func:`rotate_score <torchdrug.layers.functional.rotate_score>`,
   :class:`RotatE <torchdrug.models.RotatE>`

7. `Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs <KBGAT_>`_

   Deepak Nathani, Jatin Chauhan, Charu Sharma, Manohar Kaul. ACL 2019.

   :class:`KnowledgeBaseGraphAttentionNetwork <torchdrug.models.KnowledgeBaseGraphAttentionNetwork>`