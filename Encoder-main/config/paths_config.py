dataset_paths = {
   'deepfashion_train':'/media/hdu/eabb22ba-e327-4347-a51f-d05900de90b9/yqm/data/deep-e4e/train',
   'deepfashion_test':'/media/hdu/eabb22ba-e327-4347-a51f-d05900de90b9/yqm/data/deep-e4e/test',
   'fashiondata_train':'/media/hdu/eabb22ba-e327-4347-a51f-d05900de90b9/yqm/data/psp_fashiondata/train_img',
    'fashiondata_test':'/media/hdu/eabb22ba-e327-4347-a51f-d05900de90b9/yqm/data/psp_fashiondata//test_img',
}

model_paths = {
	#'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
        #'stylegan_ffhq': 'pretrained_models/deepfashion320.pt',
        'stylegan_ffhq': '/media/hdu/eabb22ba-e327-4347-a51f-d05900de90b9/yqm/new_ours_sematicstylegan_fashion_512_change_latent_code_size/ckpt/155000.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
