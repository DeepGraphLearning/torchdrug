import os
import unittest

import torch
from torch import multiprocessing as mp

from torchdrug.utils import comm


def worker(rank, reduce_fn, objs, queue, event):
    comm.init_process_group("nccl", init_method="env://", rank=rank)
    result = reduce_fn(objs[rank])
    queue.put((rank, result))
    event.wait()


class ReduceTest(unittest.TestCase):

    def setUp(self):
        self.num_worker = 4
        self.asymmetric_objs = []
        self.objs = []
        for i in range(self.num_worker):
            obj = {"a": torch.randint(5, (3,)).cuda(), "b": torch.rand(5).cuda()}
            asymmetric_obj = {"a": torch.randint(5, (i + 1,)).cuda(), "b": torch.rand(i * 3 + 1).cuda()}
            self.objs.append(obj)
            self.asymmetric_objs.append(asymmetric_obj)

        self.ctx = mp.get_context("spawn")
        os.environ["WORLD_SIZE"] = str(self.num_worker)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1024"

    def test_reduce(self):
        queue = self.ctx.Queue()
        event = self.ctx.Event()

        spawn_ctx = mp.spawn(worker, (comm.reduce, self.objs, queue, event), nprocs=self.num_worker, join=False)
        truth = {}
        truth["a"] = torch.stack([obj["a"] for obj in self.objs]).sum(dim=0)
        truth["b"] = torch.stack([obj["b"] for obj in self.objs]).sum(dim=0)
        for i in range(self.num_worker):
            rank, result = queue.get()
            self.assertTrue(torch.allclose(result["a"], truth["a"]), "Incorrect reduce operator")
            self.assertTrue(torch.allclose(result["b"], truth["b"]), "Incorrect reduce operator")
            del result
        event.set()
        spawn_ctx.join()

        event.clear()
        spawn_ctx = mp.spawn(worker, (comm.stack, self.objs, queue, event), nprocs=self.num_worker, join=False)
        truth = {}
        truth["a"] = torch.stack([obj["a"] for obj in self.objs])
        truth["b"] = torch.stack([obj["b"] for obj in self.objs])
        for i in range(self.num_worker):
            rank, result = queue.get()
            self.assertTrue(torch.allclose(result["a"], truth["a"]), "Incorrect stack operator")
            self.assertTrue(torch.allclose(result["b"], truth["b"]), "Incorrect stack operator")
            del result
        event.set()
        spawn_ctx.join()

        event.clear()
        spawn_ctx = mp.spawn(worker, (comm.cat, self.asymmetric_objs, queue, event), nprocs=self.num_worker, join=False)
        truth = {}
        truth["a"] = torch.cat([obj["a"] for obj in self.asymmetric_objs])
        truth["b"] = torch.cat([obj["b"] for obj in self.asymmetric_objs])
        for i in range(self.num_worker):
            rank, result = queue.get()
            self.assertTrue(torch.allclose(result["a"], truth["a"]), "Incorrect cat operator")
            self.assertTrue(torch.allclose(result["b"], truth["b"]), "Incorrect cat operator")
            del result
        event.set()
        spawn_ctx.join()


if __name__ == "__main__":
    unittest.main()