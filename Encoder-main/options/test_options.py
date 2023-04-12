from argparse import ArgumentParser

class TestOptions:
    def __init__(self):
        self.parser=ArgumentParser()
        self.initialize()

    def initialize(self):
        #path
        self.parser.add_argument('--exp_dir',type=str,help="Path to output ")
        self.parser.add_argument('--ckpt',type=str,default=None)
        self.parser.add_argument('--train_path',type=str,default=None)
        self.parser.add_argument('--test_path',type=str,default=None)

        self.parser.add_argument('--batch_size',type=int, default=2)
        self.parser.add_argument('--train_work',type=int,default=2)
        self.parser.add_argument('--test_batch',type=int,default=1)
        #self.parser.add_argument()

    def parse(self):
        opts=self.parser.parse_args()
        return opts