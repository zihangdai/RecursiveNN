import copy

from core.recnn import recnn_trainer, init_parameter
from core.datasets import entity

data = entity.load_dataset('stanford_sentiment')
params = init_parameter.init_RecNN_parameters(50,30000,5)
trainer = recnn_trainer.RecNNTrainer(50,30000,5)

test_data = data[:100]
c, g = trainer.train(params,test_data)

epsilon = 1e-4
for i in range(50*30000,params.shape[0],1):
	params_l = copy.copy(params)
	params_h = copy.copy(params)
	params_l[i] -= epsilon
	params_h[i] += epsilon
	c_l, g_l = trainer.train(params_l,test_data)
	c_h, g_h = trainer.train(params_h,test_data)
	g_h_l = (c_h - c_l)/(2*epsilon)
	print  (g_h_l-g[i]) < 1e-8, g_h_l, g[i], params_l[i], params_h[i]

#trainer.train(params, data[:100])