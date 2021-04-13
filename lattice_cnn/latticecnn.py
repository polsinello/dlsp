import networkx as nx

import numpy as np
from collections import defaultdict
from itertools import chain, combinations

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from opt_einsum import contract_expression


def compute_closure(G, R):
  def recursion(node, pred, N_rev, C):
    C[node] = C[pred].union(C[node])
    for n in N_rev[node]:
      recursion(n, node, N_rev, C)

  C = {}
  N_rev = defaultdict(set)

  for u, v in G.edges():
    N_rev[v].add(u)
  
  for n in G.nodes():
    C[n] = {n}

  recursion(R, R, N_rev, C)
  return C


class MeetSemiLattice():
  def __init__(self, L, ordering):
    self.L = L
    self.ordering = ordering
    root = [n for n, d in L.out_degree() if d < 1][0]

    self.closure = compute_closure(L, root)
    self.DLT_inv = np.zeros((L.number_of_nodes(), L.number_of_nodes()))


    for n, closure in self.closure.items():
      for m in closure:
        self.DLT_inv[self.ordering[m], self.ordering[n]] = 1

    self.DLT = np.linalg.inv(self.DLT_inv)


class JoinSemiLattice():
  def __init__(self, L, ordering):
    self.L = L
    self.ordering = ordering
    root = [n for n, d in L.in_degree() if d < 1][0]

    self.closure = compute_closure(L.reverse, root)
    self.DLT_inv = np.zeros((L.number_of_nodes(), L.number_of_nodes()))


    for n, closure in self.closure.items():
      for m in closure:
        self.DLT_inv[self.ordering[n], self.ordering[m]] = 1

    self.DLT = np.linalg.inv(self.DLT_inv)


class MeetConv(nn.Module):
  def __init__(self, L, in_channels=1, out_channels=1, bias=True, irreducibles=[], device=None):
    super().__init__()
    self.has_bias = bias
    self.expr_shape = None
    self.size = L.L.number_of_nodes()
    self.DLT = Variable(torch.Tensor(L.DLT))
    self.DLT_inv = Variable(torch.Tensor(L.DLT_inv))
    
    self.mask = Variable(torch.Tensor(irreducibles))
    #self.mask = torch.zeros(self.size)
    #self.mask[[0] + [ordering[e[0]] for e in MFO_500.in_edges('GO:0003674')]] = 1

    self.filter = nn.Parameter(torch.empty(in_channels, out_channels, self.size))
    nn.init.kaiming_normal_(self.filter, a=0.01)
    
    if self.has_bias:
      self.bias = nn.Parameter(torch.empty(out_channels, self.size))
      nn.init.kaiming_normal_(self.bias, a=0.01)

    if device is not None:
      self.DLT = self.DLT.to(device)
      self.mask = self.mask.to(device)
      self.DLT_inv = self.DLT_inv.to(device)

  def forward(self, x):
    if not self.expr_shape == x.shape:
      self.expr_shape = x.shape
      self.expr = contract_expression('nm,com,ml,bcl->bon', self.DLT_inv, self.filter.shape, self.DLT, x.shape, constants=[0, 2])

    #filt_fourier = (self.DLT_inv.T @ torch.reshape(self.filter * self.mask, (self.size, -1))).reshape(self.filter.shape)
    
    if len(irreducibles.empty()):
      x = self.expr((self.DLT_inv.T @ torch.reshape(self.filter * self.mask, (self.size, -1))).reshape(self.filter.shape), x, backend='torch')
    else:
      x = self.expr(self.filter, x, backend='torch')

    return x + (self.bias if self.has_bias else 0)


class JoinConv(nn.Module):
  def __init__(self, L, in_channels=1, out_channels=1, bias=True, device=None):
    super().__init__()
    self.has_bias = bias
    self.expr_shape = None
    self.size = L.L.number_of_nodes()
    self.DLT = Variable(torch.Tensor(L.DLT))
    self.DLT_inv = Variable(torch.Tensor(L.DLT_inv))

    self.mask = Variable(torch.Tensor(irreducibles))

    self.filter = nn.Parameter(torch.empty(in_channels, out_channels, self.size))
    nn.init.kaiming_normal_(self.filter, a=0.01)
    
    if self.has_bias:
      self.bias = nn.Parameter(torch.empty(out_channels, self.size))
      nn.init.kaiming_normal_(self.bias, a=0.01)

    if device is not None:
      self.DLT = self.DLT.to(device)
      self.mask = self.mask.to(device)
      self.DLT_inv = self.DLT_inv.to(device)

  def forward(self, x):
    if not self.expr_shape == x.shape:
      self.expr_shape = x.shape
      self.expr = contract_expression('nm,com,ml,bcl->bon', self.DLT_inv, self.filter.shape, self.DLT, x.shape, constants=[0, 2])

    if len(irreducibles.empty()):
      x = self.expr((self.DLT_inv.T @ torch.reshape(self.filter * self.mask, (self.size, -1))).reshape(self.filter.shape), x, backend='torch')
    else:
      x = self.expr(self.filter, x, backend='torch')

    return x + (self.bias if self.has_bias else 0)
