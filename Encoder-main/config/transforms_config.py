from abc import abstractmethod
import torchvision.transforms as transforms


class TransformsConfig(object):

	def __init__(self, opts):
		self.opts = opts

	@abstractmethod
	def get_transforms(self):
		pass



# class EncodeTransforms(TransformsConfig):
#
# 	def __init__(self, opts):
# 		super(EncodeTransforms, self).__init__(opts)
#
# 	def get_transforms(self):
# 		transforms_dict = {
# 			'transform_gt_train': transforms.Compose([
# 				transforms.Resize((256, 256)),
# 				transforms.RandomHorizontalFlip(0.5),
# 				transforms.ToTensor(),
# 				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
# 			'transform_source': None,
# 			'transform_test': transforms.Compose([
# 				transforms.Resize((256, 256)),
# 				transforms.ToTensor(),
# 				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
# 			'transform_inference': transforms.Compose([
# 				transforms.Resize((256, 256)),
# 				transforms.ToTensor(),
# 				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# 		}
# 		return transforms_dict


# 因为 psp具有不同应用场景因此对于着不同的场景都使用了不同transforms 类
class EncodeTransforms(TransformsConfig):

	def __init__(self, opts):
		super(EncodeTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((512, 512)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source':transforms.Compose([
				transforms.Resize((512, 512)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			# 那应该 from_image 与 to_image应该是一个；
			'transform_test': transforms.Compose([
				transforms.Resize((512, 512)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((512, 512)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_segm': transforms.Compose([
				transforms.Resize((512, 512))])
		}
		return transforms_dict

