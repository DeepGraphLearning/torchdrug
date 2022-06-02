Customize Models & Tasks
========================

TorchDrug provides many popular model architectures for graph representation
learning. However, you may still find yourself in need of some more customized
architectures.

Here we illustrate the steps for writing customized models based on the example
of `variational graph auto encoder`_ (VGAE). VGAE learns latent node representations
with a graph convolutional network (GCN) encoder and an inner product decoder.
They are jointly trained with a reconstruction loss and evaluated on the link
prediction task.

.. _variational graph auto encoder: https://arxiv.org/pdf/1611.07308.pdf

As a convention, we separate representation models and task-specific designs for
better reusability.

Node Representation Model
-------------------------

In VGAE, the node representation model is a variational graph convolutional network
(VGCN). This can be implemented via standard graph convolution layers, plus a
variational regularization loss. We define our model as a subclass of `nn.Module`
and :class:`core.Configurable <torchdrug.core.Configurable>`.

.. code:: python

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils import data as torch_data

    from torchdrug import core, layers, datasets, metrics
    from torchdrug.core import Registry as R

    @R.register("models.VGCN")
    class VariationalGraphConvolutionalNetwork(nn.Module, core.Configurable):

        def __init__(self, input_dim, hidden_dims, beta=0, batch_norm=False,
                     activation="relu"):
            super(VariationalGraphConvolutionalNetwork, self).__init__()
            self.input_dim = input_dim
            self.output_dim = hidden_dims[-1]
            self.dims = [input_dim] + list(hidden_dims)
            self.beta = beta

            self.layers = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.layers.append(
                    layers.GraphConv(self.dims[i], self.dims[i + 1], None,
                                     batch_norm, activation)
                )
            self.layers.append(
                layers.GraphConv(self.dims[-2], self.dims[-1] * 2, None, False, None)
            )

The definition is similar to most other ``torch`` models, except two points.
First, the decoration line ``@R.register("models.VGCN")`` registers the model in
the library with the name ``models.VGCN``. This enables the model to be dumped
into string format and reconstructed later. Second, ``self.input_dim`` and
``self.output_dim`` are set to inform other models that connect to it.

Then we implement the forward function. The forward function takes 4 arguments,
graph(s), node input feature(s), the global loss and the global metric. The advanatage
of these global variables is that they enable implementation of losses in a
distributed, module-centric manner.

We compute the variational regularization loss, and add it to the global loss and the
global metric.

.. code::

        def reparameterize(self, mu, log_sigma):
            if self.training:
                z = mu + torch.rand_like(mu) * log_sigma.exp()
            else:
                z = mu
            return z

        def forward(self, graph, input, all_loss=None, metric=None):
            x = input
            for layer in self.layers:
                x = layer(graph, x)
            mu, log_sigma = x.chunk(2, dim=-1)
            node_feature = self.reparameterize(mu, log_sigma)

            if all_loss is not None and self.beta > 0:
                loss = 0.5 * (mu ** 2 + log_sigma.exp() ** 2 - 2 * log_sigma - 1)
                loss = loss.sum(dim=-1).mean()
                all_loss += loss * self.beta
                metric["variational regularization loss"] = loss

            return {
                "node_feature": node_feature
            }

Here we explicitly return a dict to indicate the type of our representations. The
dict may also contain other representations, such edge representations or graph
representations.

Link Prediction Task
--------------------

Here we show how to implement the link prediction task for VGAE.

Generally, a task in TorchDrug contains 4 functions, ``predict()``, ``target()``,
``forward`` and ``evaluate()``. Such interfaces empower us to seamlessly switch
between different devices, such as CPUs, GPUs or even the distributed setting.

Among the above functions, ``predict()`` and ``target()`` compute the prediction and
the ground truth for a batch respectively. ``forward()`` compute the training loss,
while ``evaluate()`` compute the evaluation metrics.

Optionally, one can also implement ``preprocess()`` function, which performs
arbitrary operations based on the dataset.

In the case of VGAE, we first compute the undirected training graph in
``preprocess()``. In ``predict()``, we perform negative sampling, and predict
the logits for both positive and negative edges. In ``target()``, we return
the ground truth label for edges. ``evaluate()`` computes the area under ROC curve
for the predictions.

.. code:: python

    @R.register("tasks.LinkPrediction")
    class LinkPrediction(tasks.Task, core.Configurable):

        def __init__(self, model):
            super(LinkPrediction, self).__init__()
            self.model = model

        def preprocess(self, train_set, valid_set, test_set):
            dataset = train_set.dataset
            graph = dataset.graph
            train_graph = dataset.graph.edge_mask(train_set.indices)

            # flip the edges to make the graph undirected
            edge_list = train_graph.edge_list.repeat(2, 1)
            edge_list[train_graph.num_edge:, :2] = edge_list[train_graph.num_edge:, :2] \
                                                   .flip(1)
            index = torch.arange(train_graph.num_edge, device=self.device) \
                    .repeat(2, 1).t().flatten()
            data_dict, meta_dict = train_graph.data_mask(edge_index=index)
            train_graph = type(train_graph)(
                edge_list, edge_weight=train_graph.edge_weight[index],
                num_node=train_graph.num_node, num_edge=train_graph.num_edge * 2,
                meta_dict=meta_dict, **data_dict
            )

            self.register_buffer("train_graph", train_graph)
            self.num_node = dataset.num_node

        def forward(self, batch):
            all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            metric = {}

            pred = self.predict(batch, all_loss, metric)
            target = self.target(batch)
            metric.update(self.evaluate(pred, target))

            loss = F.binary_cross_entropy_with_logits(pred, target)
            metric["bce loss"] = loss

            all_loss += loss

            return all_loss, metric

        def predict(self, batch, all_loss=None, metric=None):
            neg_batch = torch.randint(self.num_node, batch.shape, device=self.device)
            batch = torch.cat([batch, neg_batch])
            node_in, node_out = batch.t()

            output = self.model(self.train_graph, self.train_graph.node_feature.float(),
                                all_loss, metric)
            node_feature = output["node_feature"]
            pred = torch.einsum("bd, bd -> b",
                                node_feature[node_in], node_feature[node_out])
            return pred

        def target(self, batch):
            batch_size = len(batch)
            target = torch.zeros(batch_size * 2, device=self.device)
            target[:batch_size] = 1
            return target

        def evaluate(self, pred, target):
            roc = metrics.area_under_roc(pred, target)
            return {
                "AUROC": roc
            }

Put Them Together
-----------------

Let's put all the ingredients together. Since the original Cora is a node
classification dataset, we apply a wrapper to make it compatible with link
prediction.

.. code:: python

    class CoraLinkPrediction(datasets.Cora):

        def __getitem__(self, index):
            return self.graph.edge_list[index]

        def __len__(self):
            return self.graph.num_edge

    dataset = CoraLinkPrediction("~/node-datasets/")
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch_data.random_split(dataset, lengths)

    model = VariationalGraphConvolutionalNetwork(dataset.node_feature_dim, [128, 16],
                                                 beta=1e-3, batch_norm=True)
    task = LinkPrediction(model)

    optimizer = torch.optim.Adam(task.parameters(), lr=1e-2)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0],
                         batch_size=len(train_set))
    solver.train(num_epoch=200)
    solver.evaluate("valid")

The result may look like

.. code:: bash

    AUROC: 0.898589