"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import  sys
sys.path.append('.')
sys.path.append("/home/hdu/yqm/Encoder/Encoder-main")
import math
import os
import torch
from torch import nn
from models import make_model
from models.EE import EE_Encoder
from models.semantic_stylegan import SemanticGenerator as Generator
from config.paths_config import model_paths
from feature_modulation import LocalFeatEncoder
def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt
class EEncoder(nn.Module):
	def __init__(self, opts):
		super(EEncoder, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.featEncoder=self.set_featEncoder()
		self.decoder=self.set_decoder()
		#pSp可以看作是encoder - decoder  decoder 部分就是 generator ，
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		#这个顺序 （1） 创建网络 （2）load 权重
		self.load_weights()
	def set_featEncoder(self):
		featEncoder=LocalFeatEncoder()
		return self.featEncoder.to(self.opts.device)

	def set_decoder(self):
		self.ckpt = torch.load(self.opts.ckpt)
		self.ckpt['args'].num_workers = 1
		self.decoder = make_model(self.ckpt['args'])
		return self.decoder.to(self.opts.device)

	def set_encoder(self):
		#写成这样应该跟消融实验相关。
		encoder = EE_Encoder(self.opts.output_size, 64).to(self.opts.device)
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			# 这个位置的代码该如何进行修改
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.ckpt)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			self.__load_latent_avg(ckpt, repeat=1)



# 在predict.py中会调用这个 forward
	def forward(self, x, resize=False, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		#
		if input_code:
			codes = x
			#注意这里是 net 相当于  pSp = encoder+ decoder
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		if latent_mask is not None:
			for i in latent_mask:
				# 将要进行风格混合的w 以list形式给出来,例如[8,9,10,11,12,13,14，15，16，17] 将这些层进行混合。
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0
		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)
		# 得到 res_image 计算delta_image ,得到对应的 mean 以及方差
		imgs_ = torch.nn.functional.interpolate(torch.clamp(images, -1., 1.), size=(256, 256), mode='bilinear')
		delta_image=(x-images).detach() # other should not  train
		conditions=self.featEncoder(delta_image)
		if conditions  is not None:
			images, result_latent = self.decoder([codes],
												 input_is_latent=input_is_latent,
												 randomize_noise=randomize_noise,
												 return_latents=return_latents,conditions=conditions)

		if return_latents:
			return images, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
