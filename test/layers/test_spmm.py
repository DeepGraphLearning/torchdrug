import unittest

from itertools import product

import torch
import torch_scatter

from torchdrug import data, utils
from torchdrug.layers import functional


class SPMMTest(unittest.TestCase):

    def setUp(self):
        self.num_node = 50
        self.num_relation = 10
        self.dim = 20
        adjacency = torch.rand(self.num_node, self.num_node)
        threshold = adjacency.flatten().kthvalue((self.num_node - 10) * self.num_node)[0]
        adjacency = adjacency * (adjacency > threshold)
        self.graph = data.Graph.from_dense(adjacency)
        rel_adjacency = torch.rand(self.num_node, self.num_node, self.num_relation)
        threshold = rel_adjacency.flatten().kthvalue((self.num_node - 10) * self.num_node)[0]
        rel_adjacency = rel_adjacency * (rel_adjacency > threshold)
        self.knowledge_graph = data.Graph.from_dense(rel_adjacency)
        self.relation = torch.rand(self.num_relation, self.dim)
        self.input = torch.rand(self.num_node, self.dim)
        self.output_grad = torch.rand(self.num_node, self.dim)
        self.operators = [("add", "mul"), ("min", "mul"), ("max", "mul"), ("min", "add"), ("max", "add")]
        self.devices = ["CPU", "CUDA"]

    def test_spmm(self):
        for device, (sum_op, mul_op) in product(self.devices, self.operators):
            if device == "CUDA":
                self.graph = self.graph.cuda()
                self.input = self.input.cuda()
                self.output_grad = self.output_grad.cuda()
            self.graph.edge_weight.requires_grad_()
            self.input.requires_grad_()

            node_in, node_out = self.graph.edge_list.t()
            result = functional.generalized_spmm(self.graph.adjacency.t(), self.input, sum=sum_op, mul=mul_op)
            sum_func = getattr(torch_scatter, "scatter_%s" % sum_op)
            mul_func = getattr(torch, mul_op)
            edge_weight = self.graph.edge_weight.unsqueeze(-1)
            message = mul_func(edge_weight, self.input[node_in])
            truth = sum_func(message, node_out, dim=0, dim_size=self.num_node)
            if isinstance(truth, tuple):
                truth = truth[0]
            self.assertTrue(torch.allclose(result, truth),
                            "Incorrect generalized spmm forward (sum=`%s`, mul=`%s`)" % (sum_op, mul_op))

            result_edge, result_input = torch.autograd.grad(
                result, (self.graph.edge_weight, self.input), self.output_grad)
            truth_edge, truth_input = torch.autograd.grad(
                truth, (self.graph.edge_weight, self.input), self.output_grad)
            self.assertTrue(torch.allclose(result_edge, truth_edge),
                            "Incorrect generalized spmm backward (sum=`%s`, mul=`%s`)" % (sum_op, mul_op))
            self.assertTrue(torch.allclose(result_input, truth_input),
                            "Incorrect generalized spmm backward (sum=`%s`, mul=`%s`)" % (sum_op, mul_op))

    def test_rspmm(self):
        for device, (sum_op, mul_op) in product(self.devices, self.operators):
            if device == "CUDA":
                self.knowledge_graph = self.knowledge_graph.cuda()
                self.relation = self.relation.cuda()
                self.input = self.input.cuda()
                self.output_grad = self.output_grad.cuda()
            self.knowledge_graph.edge_weight.requires_grad_()
            self.relation.requires_grad_()
            self.input.requires_grad_()

            result = functional.generalized_rspmm(self.knowledge_graph.adjacency.transpose(0, 1),
                                                  self.relation, self.input, sum=sum_op, mul=mul_op)
            sum_func = getattr(torch_scatter, "scatter_%s" % sum_op)
            mul_func = getattr(torch, mul_op)
            node_in, node_out, relation = self.knowledge_graph.edge_list.t()
            edge_weight = self.knowledge_graph.edge_weight.unsqueeze(-1)
            message = mul_func(self.relation[relation], self.input[node_in])
            truth = sum_func(edge_weight * message, node_out, dim=0, dim_size=self.num_node)
            if isinstance(truth, tuple):
                truth = truth[0]
            self.assertTrue(torch.allclose(result, truth),
                            "Incorrect generalized rspmm forward (sum=`%s`, mul=`%s`)" % (sum_op, mul_op))

            result_edge, result_relation, result_input = torch.autograd.grad(
                result, (self.knowledge_graph.edge_weight, self.relation, self.input), self.output_grad)
            truth_edge, truth_relation, truth_input = torch.autograd.grad(
                truth, (self.knowledge_graph.edge_weight, self.relation, self.input), self.output_grad)
            self.assertTrue(torch.allclose(result_edge, truth_edge),
                            "Incorrect generalized rspmm backward (sum=`%s`, mul=`%s`)" % (sum_op, mul_op))
            self.assertTrue(torch.allclose(result_relation, truth_relation),
                            "Incorrect generalized rspmm backward (sum=`%s`, mul=`%s`)" % (sum_op, mul_op))
            self.assertTrue(torch.allclose(result_input, truth_input),
                            "Incorrect generalized rspmm backward (sum=`%s`, mul=`%s`)" % (sum_op, mul_op))


if __name__ == "__main__":
    unittest.main()