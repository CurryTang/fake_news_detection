import torch_geometric as tg
import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, GINConv, global_add_pool, global_sort_pool
from torch_geometric.nn.norm import BatchNorm
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
import copy as cp 
import torch.nn.functional as F
from utils import enhance_root_edge_index_ud, enhance_root_edge_index, dropedge
from torch_geometric.utils import softmax, is_undirected, to_undirected
from torch_scatter import scatter_add
import numpy as np


class GlobalAttentionPool(torch.nn.Module):
	def __init__(self, num_in, num_hidden):
		super().__init__()
		# self.fc = Linear(num_in * 2, 1)
		# self.act = torch.nn.LeakyReLU(negative_slope = slope)
		self.att = torch.nn.Sequential(
		  torch.nn.Linear(num_in * 2, num_hidden),
		  torch.nn.Tanh(),
		  torch.nn.Linear(num_hidden, 1, bias = False)
		)
		# self.att = torch.nn.Linear(num_in, num_in, bias = False)

	def forward(self, x, batch):
		root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
		root = torch.cat([root.new_zeros(1), root + 1], dim=0)
		root_feature = x[root]
		repeat_num = root[1:]
		repeat_num_shift = torch.diff(repeat_num)
		repeat_num_shift = torch.cat([repeat_num_shift.new_zeros(1).fill_(repeat_num[0]), repeat_num_shift, repeat_num_shift.new_zeros(1).fill_((x.shape[0] - repeat_num[-1]))], dim = 0)
		root_feature_full = torch.repeat_interleave(root_feature, repeat_num_shift, dim = 0)
		concat_feature = torch.cat([x, root_feature_full], dim = -1)
		# beta = torch.tanh(self.fc(concat_feature))
		# beta = self.act(self.fc(concat_feature))
		# x_transform = self.att(x).unsqueeze(-1)
		# root_feature_full = root_feature_full.unsqueeze(1)
		# beta = torch.tanh(torch.bmm(root_feature_full, x_transform)).view(-1, 1)
		beta = self.att(concat_feature)
		alpha = softmax(beta, batch).view(-1, 1)
		return scatter_add(alpha * x, batch, dim = 0)





pooling_dict = {
	'max': global_max_pool,
	'mean': global_mean_pool,
	'add': global_add_pool,
	'sort': global_sort_pool,
	'attention': GlobalAttentionPool
}





class MLP(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels):
		super().__init__()
		self.lin0 = Linear(in_channels, hidden_channels)
		self.lin1 = Linear(hidden_channels, out_channels)
	
	def forward(self, x):
		h = self.lin0(x).relu()
		h = self.lin1(h)
		return h.log_softmax(dim = -1)

class Net(torch.nn.Module):
	def __init__(self, model, in_channels, hidden_channels, out_channels,
								concat=True):
		super().__init__()
		self.concat = concat

		if model == 'GCN' or model == 'GDC':
				self.conv1 = GCNConv(in_channels, hidden_channels)
		elif model == 'SAGE':
				self.conv1 = SAGEConv(in_channels, hidden_channels)
		elif model == 'GAT':
				self.conv1 = GATConv(in_channels, hidden_channels)

		if self.concat:
				self.lin0 = Linear(in_channels, hidden_channels)
				self.lin1 = Linear(2 * hidden_channels, hidden_channels)

		self.lin2 = Linear(hidden_channels, out_channels)

	def forward(self, x, edge_index, batch):
		edge_index = to_undirected(edge_index)
		h = self.conv1(x, edge_index).relu()
		h = global_max_pool(h, batch)

		if self.concat:
				# Get the root node (tweet) features of each graph:
				root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
				root = torch.cat([root.new_zeros(1), root + 1], dim=0)
				news = x[root]

				news = self.lin0(news).relu()
				h = self.lin1(torch.cat([news, h], dim=-1)).relu()

		h = self.lin2(h)
		return h.log_softmax(dim=-1), None

class UDNewsNet(torch.nn.Module):
	def __init__(self, num_features, num_hidden, num_classes, need_jk = False, pooling = 'max', enhance = True, mixup = False, mixup_alpha = 1.0, att_hidden = 64):
		super().__init__()
		self.conv_1 = GCNConv(num_features, num_hidden)
		self.conv_2 = GCNConv(num_features + num_hidden, num_hidden)
		# self.conv_2 = GCNConv(num_hidden, num_hidden)
		#self.linear = Linear(num_hidden, num_classes)
		self.root_linear = Linear(num_features, num_hidden)
		self.jk = need_jk 
		if not self.jk:
			self.linear = Linear(2 * num_hidden, num_classes)
			# self.linear = Linear(num_hidden, num_classes)
		else:
			self.linear = Linear(3 * num_hidden, num_classes)
		self.pooling = pooling
		## pooling num for sortPooling
		self.pooling_num = 1
		## need root edge enhancement or not
		self.enhance = enhance
		## mixup technique
		self.mixup = mixup
		if self.mixup:
			self.mixup_alpha = mixup_alpha
			self.lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
		# if self.virtual_node:
		#   self.vn_emb = torch.nn.Embedding(1, num_hidden)
		#   torch.nn.init.constant_(self.vn_emb.weight.data, 0)
		if self.pooling == 'attention':
			self.att_hidden = att_hidden
			self.att_pool_1 = GlobalAttentionPool(num_hidden, self.att_hidden)
			self.att_pool_2 = GlobalAttentionPool(num_hidden, self.att_hidden)
	def forward(self, x, edge_index, batch):
		edge_index = to_undirected(edge_index)
		if self.enhance:
			edge_index = enhance_root_edge_index(x, edge_index, batch)
		root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
		root = torch.cat([root.new_zeros(1), root + 1], dim=0)
		root_feature = x[root] 
		repeat_num = root[1:]
		repeat_num_shift = torch.diff(repeat_num)
		repeat_num_shift = torch.cat([repeat_num_shift.new_zeros(1).fill_(repeat_num[0]), repeat_num_shift, repeat_num_shift.new_zeros(1).fill_((x.shape[0] - repeat_num[-1]))], dim = 0)
		root_feature_full = torch.repeat_interleave(root_feature, repeat_num_shift, dim = 0)
		# print(edge_index.shape)
		# input()
		layer1_out = self.conv_1(x, edge_index).relu()
		layer1_out = F.dropout(layer1_out, training = self.training)
		if self.pooling == 'max':
			layer1_pool = global_max_pool(layer1_out, batch)
		elif self.pooling == 'mean':
			layer1_pool = global_mean_pool(layer1_out, batch)
		elif self.pooling == 'add':
			layer1_pool = global_add_pool(layer1_out, batch)
		elif self.pooling == 'sort':
			layer1_pool = global_sort_pool(layer1_out, batch, self.pooling_num)
		else:
			layer1_pool = self.att_pool_1(layer1_out, batch)
		layer2_input = torch.cat([x, layer1_out], dim = -1)
		# layer2_input = layer1_out
		layer2_output = self.conv_2(layer2_input, edge_index)
		layer2_output = F.dropout(layer2_output,training = self.training)
		if self.pooling == 'max':
			result = global_max_pool(layer2_output, batch)
		elif self.pooling == 'mean':
			result = global_mean_pool(layer2_output, batch)
		elif self.pooling == 'add':
			result = global_add_pool(layer2_output, batch)
		elif self.pooling == 'sort':
			result = global_sort_pool(layer2_output, batch, self.pooling_num)
		else:
			result = self.att_pool_2(layer2_output, batch)
		# result = global_max_pool(layer2_output, batch)
		root_feature_trans = self.root_linear(root_feature).relu()
		if self.jk:
			total = torch.cat([root_feature_trans, result, layer1_pool], dim = -1)
		else:
			total = torch.cat([root_feature_trans, result], dim = -1)
			# total = result
		# total = result
		idx = torch.randperm(total.shape[0])
		total_mix_up = total[idx, :]
		if self.mixup:
			total = self.lam * total + (1 - self.lam) * total_mix_up
		output = self.linear(total)
		return output.log_softmax(dim = -1), idx

class NewsNet(torch.nn.Module):
	def __init__(self, num_features, num_hidden, num_classes):
		super().__init__()
		self.bottom_up_conv_1 = tg.nn.GCNConv(num_features, num_hidden)
		self.top_down_conv_1 = tg.nn.GCNConv(num_features, num_hidden)
		self.root_linear = Linear(num_features, num_hidden)
		# self.top_down_linear = Linear(num_features, num_hidden)
		self.bottom_up_conv_2 = tg.nn.GCNConv(num_hidden + num_features, num_hidden)
		self.top_down_conv_2 = tg.nn.GCNConv(num_hidden + num_features, num_hidden)
		self.linear = Linear(num_hidden * 2, num_classes)

	def forward(self, x, edge_index, batch):
		top_down_edge_index = edge_index
		row, col = top_down_edge_index
		bottom_up_edge_index = torch.tensor([col.tolist(), row.tolist()]).to(x.device)
		root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
		root = torch.cat([root.new_zeros(1), root + 1], dim=0)
		root_feature = x[root] 
		repeat_num = root[1:]
		repeat_num_shift = torch.diff(repeat_num)
		repeat_num_shift = torch.cat([repeat_num_shift.new_zeros(1).fill_(repeat_num[0]), repeat_num_shift, repeat_num_shift.new_zeros(1).fill_((x.shape[0] - repeat_num[-1]))], dim = 0)
		root_feature_full = torch.repeat_interleave(root_feature, repeat_num_shift, dim = 0)
		top_down_1 = self.bottom_up_conv_1(x, top_down_edge_index)
		bottom_up_1 = self.top_down_conv_1(x, bottom_up_edge_index)
		root_1 = self.root_linear(root_feature_full)
		top_down_input = F.dropout(torch.cat([top_down_1, root_feature_full], dim = -1).relu())
		bottom_up_input = F.dropout(torch.cat([bottom_up_1, root_feature_full], dim = -1).relu())
		top_down_2 = self.bottom_up_conv_2(top_down_input, top_down_edge_index).relu()
		bottom_up_2 = self.top_down_conv_2(bottom_up_input, bottom_up_edge_index).relu()
		top_down = global_mean_pool(top_down_2, batch)
		bottom_up = global_mean_pool(bottom_up_2, batch)
		# root_feature_trans = self.root_linear(root_feature).relu()
		# root_feature_trans = root_feature
		total = torch.cat([top_down, bottom_up], dim = -1)
		output = self.linear(total)
		return output.log_softmax(dim = -1)

		
class BiGCN(torch.nn.Module):
	def __init__(self, num_features, num_hidden, num_classes):
		super().__init__()
		self.bottom_up_conv_1 = tg.nn.GCNConv(num_features, num_hidden)
		self.top_down_conv_1 = tg.nn.GCNConv(num_features, num_hidden)
		self.bottom_up_conv_2 = tg.nn.GCNConv(num_features + num_hidden, num_hidden)
		self.top_down_conv_2 = tg.nn.GCNConv(num_features + num_hidden, num_hidden)
		self.linear = Linear(num_hidden * 2, num_classes)
		self.pooling = global_mean_pool

	def forward(self, x, edge_index, batch):
		top_down_edge_index = edge_index
		row, col = top_down_edge_index
		bottom_up_edge_index = torch.tensor([col.tolist(), row.tolist()]).to(x.device)
		root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
		root = torch.cat([root.new_zeros(1), root + 1], dim=0)
		root_feature = x[root] 
		repeat_num = root[1:]
		repeat_num_shift = torch.diff(repeat_num)
		repeat_num_shift = torch.cat([repeat_num_shift.new_zeros(1).fill_(repeat_num[0]), repeat_num_shift, repeat_num_shift.new_zeros(1).fill_((x.shape[0] - repeat_num[-1]))], dim = 0)
		root_feature_full = torch.repeat_interleave(root_feature, repeat_num_shift, dim = 0)
		## layer1
		bottom_up_edge_index = dropedge(bottom_up_edge_index, 0)
		top_down_edge_index = dropedge(top_down_edge_index, 0)
		layer1_bu = self.bottom_up_conv_1(x, bottom_up_edge_index)
		layer1_td = self.top_down_conv_1(x, top_down_edge_index)
		# print(1)
		layer1_top_down = self.pooling(layer1_td, batch)
		layer1_bottom_up = self.pooling(layer1_bu, batch)
		concat_bu = torch.cat([root_feature_full, layer1_bu], dim = -1)
		concat_td = torch.cat([root_feature_full, layer1_td], dim = -1)
		concat_bu = F.dropout(concat_bu.relu(), training = self.training)
		concat_td = F.dropout(concat_td.relu(), training = self.training)
		layer2_bu = self.bottom_up_conv_2(concat_bu, edge_index).relu()
		layer2_td = self.top_down_conv_2(concat_td, edge_index).relu()
		layer2_bu_head = layer2_bu[root]
		layer2_td_head = layer2_td[root]
		layer2_bu_head_full = torch.repeat_interleave(layer2_bu_head, repeat_num_shift, dim = 0)
		layer2_td_head_full = torch.repeat_interleave(layer2_td_head, repeat_num_shift, dim = 0)
		layer2_bu = self.pooling(layer2_bu_head_full, batch)
		layer2_td = self.pooling(layer2_td_head_full, batch)
		total = torch.cat([layer2_bu, layer2_td], dim = -1)        
		output = self.linear(total)
		return output.log_softmax(dim = -1), None


class GCNFN(torch.nn.Module):
	def __init__(self, num_features, num_hidden, num_classes):
		super().__init__()

		self.num_features = num_features
		self.num_classes = num_classes
		self.nhid = num_hidden

		self.conv1 = GATConv(self.num_features, self.nhid * 2)
		self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

		self.fc1 = Linear(self.nhid * 2, self.nhid)

		self.fc2 = Linear(self.nhid, self.num_classes)


	def forward(self, x, edge_index, batch):
		edge_index = to_undirected(edge_index)
		x = F.selu(self.conv1(x, edge_index))
		x = F.selu(self.conv2(x, edge_index))
		x = F.selu(global_mean_pool(x, batch))
		x = F.selu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.log_softmax(self.fc2(x), dim=-1)

		return x, None



