from argparse import ArgumentParser

class TrainOptions:
    def __init__(self):
        self.parser=ArgumentParser()
        self.initialize()

    def initialize(self):
        #path
        self.parser.add_argument('--exp_dir',type=str,help="Path to output ")
        self.parser.add_argument('--ckpt',type=str,default="/home/hdu/yqm/SemanticStyGan/pretrained/215000.pt")
        self.parser.add_argument('--train_path',type=str,default=None)
        self.parser.add_argument('--test_path',type=str,default=None)
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--train_segm', default='/media/hdu/eabb22ba-e327-4347-a51f-d05900de90b9/yqm/data/psp_fashiondata/train_label', type=str)
        self.parser.add_argument('--test_segm', default='/media/hdu/eabb22ba-e327-4347-a51f-d05900de90b9/yqm/data/psp_fashiondata/test_label', type=str)
        self.parser.add_argument('--checkpoint_path', default=None, type=str,help='trained weights')
        self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
        # network related
        # size  ,device ,optim , related 
        self.parser.add_argument('--batch_size',type=int, default=2)
        self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--train_decoder',type=bool, default=False)
        self.parser.add_argument('--train_encoder', type=bool, default=False)
        self.parser.add_argument('--train_featEncoder', type=bool, default=False)
        self.parser.add_argument('--learning_rate', type=float, default=0.0001)
        self.parser.add_argument('--train_work',type=int,default=2)
        self.parser.add_argument('--test_batch',type=int,default=1)
        self.parser.add_argument('--device', type=str, default="cuda:0")
        self.parser.add_argument('--output_size',type=int, default=512)
        self.parser.add_argument('--workers',type=int, default=2,help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers',type=int, default=2,help='Number of test/inference dataloader workers')
        self.parser.add_argument('--optim_name',type=str, default='ranger',help='Which optimizer to use')
        self.parser.add_argument('--use_wandb', default=False, type=bool, help='Whether to use Weights & Biases to track experiment')
        self.parser.add_argument('--use_full_segm',type=bool, default=True,help='')
        #hyper-lambda
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=1.0, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--l1_lambda', default=0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--style_lambda', default=0.8, type=float, help='Style loss multiplier factor')
        # other 
        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

    def parse(self):
        opts=self.parser.parse_args()
        return opts


