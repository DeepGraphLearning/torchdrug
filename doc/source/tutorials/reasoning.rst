Knowledge Graph Reasoning
=========================

.. include:: ../bibliography.rst

In knowledge graphs, one important task is knowledge graph reasoning, which aims
at predicting missing (h,r,t)-links given existing (h,r,t)-links in a knowledge
graph. There are two kinds of well-known approaches to knowledge graph reasoning.
One is knowledge graph embedding and the other one is neural inductive logic
programming.

In this tutorial, we provide two examples to illustrate how to use TorchDrug
for knowledge graph reasoning.

Knowledge Graph Embedding
-------------------------

For knowledge graph reasoning, the first kind of popular method is the knowledge
graph embedding method. The basic idea is to learn an embedding vector for each
entity and relation in a knowledge graph based on existing (h,r,t)-links. Then
these embeddings are further used to predict missing links.

Next, we will introduce how to use knowledge graph embedding models for knowledge
graph reasoning.

Prepare the Dataset
^^^^^^^^^^^^^^^^^^^

We use the `FB15k-237`_ dataset for illustration. `FB15k-237`_ is constructed from
Freebase, and the dataset has 14,541 entities as well as 237 relations. For the
dataset, there is a standard split of training/validation/test sets. We can load
the dataset using the following code:

.. code:: python

    import torch
    from torchdrug import core, datasets, tasks, models

    dataset = datasets.FB15k237("~/kg-datasets/")
    train_set, valid_set, test_set = dataset.split()

Define our Model
^^^^^^^^^^^^^^^^

Once we load the dataset, we are ready to build the model. Let's take the RotatE
model as an example, we can use the following code for model construction.

.. code:: python

    model = models.RotatE(num_entity=dataset.num_entity,
                          num_relation=dataset.num_relation,
                          embedding_dim=2048, max_score=9)

Here, ``embedding_dim`` specifies the dimension of entity and relation embeddings.
``max_score`` specifies the bias for inferring the plausibility of a (h,r,t)
triplet.

You may consider using a smaller embedding dimension for better efficiency.

Afterwards, we further need to define our task. For the knowledge graph embedding
task, we can simply use the following code.

.. code:: python

    task = tasks.KnowledgeGraphCompletion(model, num_negative=256,
                                          adversarial_temperature=1)

Here, ``num_negative`` is the number of negative examples used for training, and
``adversarial_temperature`` is the temperature for sampling negative examples.

Train and Test
^^^^^^^^^^^^^^

Afterwards, we can now train and test our model. For model training, we need to
set up an optimizer and put everything together into an Engine instance with the
following code.

.. code:: python

    optimizer = torch.optim.Adam(task.parameters(), lr=2e-5)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                         gpus=[0], batch_size=1024)
    solver.train(num_epoch=200)

Here, we can reduce ``num_epoch`` for better efficiency.

Afterwards, we may further evaluate the model on the validation set using the
following code.

.. code:: python

    solver.evaluate("valid")

Neural Inductive Logic Programming
----------------------------------

The other kind of popular method is neural inductive logic programming. The idea
of neural inductive logic programming is to learn logic rules from training data.
Once the logic rules are learned, they can be further used to predict missing links.

One popular method of neural inductive logic programming is NeuralLP. NeuralLP
considers all the chain-like rules (e.g., nationality = born_in + city_of) up to a
maximum length. Also, an attention mechanism is used to assign a scalar weight to
each logic rule. During training, the attention module is trained, so that we can
learn a proper weight for each rule. During testing, the logic rules and their
weights are used together to predict missing links.

Next, we will introduce how to deploy a NeuralLP model for knowledge graph reasoning.

Prepare the Dataset
^^^^^^^^^^^^^^^^^^^

We start with loading the dataset. Similar to the tutorial of knowledge graph
embedding, the `FB15k-237`_ dataset is used for illustration. We can load the
dataset by running the following commands:

.. code:: python

    import torch
    from torchdrug import core, datasets, tasks, models

    dataset = datasets.FB15k237("~/kg-datasets/")
    train_set, valid_set, test_set = dataset.split()

Define our Model
^^^^^^^^^^^^^^^^
Afterwards, we can now define the NeuralLP model with the following codes:

.. code:: python

    model = models.NeuralLP(num_relation=dataset.num_relation,
                            hidden_dim=128,
                            num_step=3,
                            num_lstm_layer=2)

Here, ``hidden_dim`` is the dimension of entity and relation embeddings used in
NeuralLP. ``num_step`` is the maximum length of the chain-like rules (i.e., the
maximum number of relations in the body of a chain-like rule), which is typically
set to 3. ``num_lstm_layer`` is the number of LSTM layers used in NeuralLP.

Once we define our model, we are ready to define the task. As training NeuralLP
shares similar ideas to training knowledge graph embedding, we also use the following
knowledge graph embedding task:

.. code:: python

    task = tasks.KnowledgeGraphCompletion(model, fact_ratio=0.75,
                                          num_negative=256,
                                          sample_weight=False)

The difference is that we need to specify the ``fact_ratio``, which tells the code
how many facts are used to construct the background knowledge graph on which we
perform reasoning, and this hyperparameter is typically set to 0.75.

Train and Test
^^^^^^^^^^^^^^

With the model and task we have defined, we can not perform model training and
testing. Model training is similar to that of knowledge graph embedding models,
where we need to create an optimizer and feed every component into an Engine instance
by running the following code:

.. code:: python

    optimizer = torch.optim.Adam(task.parameters(), lr=1.0e-3)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                         gpus=[0, 1, 2, 3], batch_size=64)
    solver.train(num_epoch=10)

Here, ``gpus`` specifies the GPUs on which we would like to train the model. We may
specify multiple GPUs by using the form as above. For ``num_epoch``, we can reduce
the value for efficiency purpose.

After model training, we can further use the following codes to evaluate the model
on the validation set

.. code:: python

    solver.evaluate("valid")


