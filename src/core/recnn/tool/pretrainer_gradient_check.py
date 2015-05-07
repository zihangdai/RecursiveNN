from scipy.optimize import fmin_l_bfgs_b

from core.datasets.entity import load_pretrain_dataset
from core.recnn.init_parameter import init_RecNN_parameters
from core.recnn.recnn_pretrainer import RecNNPreTrainer

data = load_pretrain_dataset('stanford_sentiment')

word_vec_dim = 50
vocabulary_size = 21702
classify_category_num = 5
epsilon = 1e-4

word_dict_end         = word_vec_dim * vocabulary_size
weights_score_end     = word_dict_end + word_vec_dim
weights_forward_end   = weights_score_end + word_vec_dim * (word_vec_dim*2+1)

params = init_RecNN_parameters(word_vec_dim,vocabulary_size,classify_category_num)
pretrainer = RecNNPreTrainer(word_vec_dim,vocabulary_size,classify_category_num)

weights_classify_end  = params.shape[0]

c, g = pretrainer.pre_train(params,data)

for i in range(0,weights_classify_end,1):
	params_l = np.array(params, dtype=np.float64)
	params_h = np.array(params, dtype=np.float64)
	params_l[i] -= epsilon
	params_h[i] += epsilon
	c_l, g_l = pretrainer.pre_train(params_l,data)
	c_h, g_h = pretrainer.pre_train(params_h,data)
	g_h_l = (c_h - c_l)/(2*epsilon)
	diff = np.abs(g_h_l - g[i])
	if not diff < 1e-9:
		print  i, diff < 1e-9, g_h_l, g[i], params_l[i], params_h[i]

for i in range(0,word_dict_end,1):
	params_l = np.array(params, dtype=np.float64)
	params_h = np.array(params, dtype=np.float64)
	params_l[i] -= epsilon
	params_h[i] += epsilon
	c_l, g_l = pretrainer.pre_train(params_l,data)
	c_h, g_h = pretrainer.pre_train(params_h,data)
	g_h_l = (c_h - c_l)/(2*epsilon)
	diff = np.abs(g_h_l - g[i])
	print  i, diff < 1e-9, g_h_l, g[i], params_l[i], params_h[i]

for i in range(word_dict_end,weights_score_end,1):
	params_l = np.array(params, dtype=np.float64)
	params_h = np.array(params, dtype=np.float64)
	params_l[i] -= epsilon
	params_h[i] += epsilon
	c_l, g_l = pretrainer.pre_train(params_l,data)
	c_h, g_h = pretrainer.pre_train(params_h,data)
	g_h_l = (c_h - c_l)/(2*epsilon)
	diff = np.abs(g_h_l - g[i])
	print  i, diff < 1e-9, g_h_l, g[i], params_l[i], params_h[i]

for i in range(weights_score_end,weights_forward_end,1):
	params_l = np.array(params, dtype=np.float64)
	params_h = np.array(params, dtype=np.float64)
	params_l[i] -= epsilon
	params_h[i] += epsilon
	c_l, g_l = pretrainer.pre_train(params_l,data)
	c_h, g_h = pretrainer.pre_train(params_h,data)
	g_h_l = (c_h - c_l)/(2*epsilon)
	diff = np.abs(g_h_l - g[i])
	print  i, diff < 1e-9, g_h_l, g[i], params_l[i], params_h[i], params[i]

for i in range(weights_forward_end,weights_classify_end,1):
	params_l = np.array(params, dtype=np.float64)
	params_h = np.array(params, dtype=np.float64)
	params_l[i] -= epsilon
	params_h[i] += epsilon
	c_l, g_l = pretrainer.pre_train(params_l,data)
	c_h, g_h = pretrainer.pre_train(params_h,data)
	g_h_l = (c_h - c_l)/(2*epsilon)
	diff = np.abs(g_h_l - g[i])
	print  i, diff < 1e-9, g_h_l, g[i], params_l[i], params_h[i], params[i]

