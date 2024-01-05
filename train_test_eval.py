import os
import torch
import Training
import Testing
from Evaluation import main
import argparse

if __name__ == "__main__":

     parser = argparse.ArgumentParser()
     # train
     parser.add_argument('--Training', default=True, type=bool, help='Training or not')
     parser.add_argument('--data_root', default='./data', type=str, help='data path')
     parser.add_argument('--train_steps', default=70000, type=int, help='total training steps')
     parser.add_argument('--img_size', default=224, type=int, help='network input size')
     # parser.add_argument('--pretrained_model', default='./pretrained_model/T2T_ViT_t_14.pth.tar', type=str, help='load Pretrained model')
     parser.add_argument('--pretrained_model_v2', default='./pretrained_model/T2T_ViT_t_14_v2.pth.tar', type=str, help='load Pretrained model')
     parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
     parser.add_argument('--stepvalue1', default=45000, type=int, help='the step 1 for adjusting lr')
     parser.add_argument('--stepvalue2', default=60000, type=int, help='the step 2 for adjusting lr')
     parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')

     # test
     parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
     parser.add_argument('--test_name', default='test2', type=str, help='test1 or test2 or test3 for PAVS')
     parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')


     # evaluation
     parser.add_argument('--Evaluation', default=True, type=bool, help='Evaluation or not')
     parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')
     
     parser.add_argument('--lr', default=0.0001, type=int, help='learning rate')
     parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
     parser.add_argument('--epochs', default=20000, type=int, help='epochs')
     parser.add_argument('--pretrained', default=True, type=bool, help='t2t pretrained or not.')
     parser.add_argument('--methods', type=str, default='DATFormer', help='evaluated method name')
     parser.add_argument('--init_method', default='tcp://127.0.0.1:33114', type=str, help='init_method')
     parser.add_argument('--trainset', default='360-SOD', type=str, help='Trainging set') # 360-SOD  F-360iSOD 360-SSOD PAVS

     parser.add_argument('--save_test_checkpoint', default='360-SOD_SODFormer45000.pth', type=str, help='checkpoint filename')
     parser.add_argument('--test_paths', type=str, default='360-SOD')
     args = parser.parse_args()

     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

     num_gpus = torch.cuda.device_count()
     print(num_gpus)

     if args.Training:
          Training.train_net(num_gpus=num_gpus, args=args)
     if args.Testing:
          Testing.test_net(args)
     if args.Evaluation:
          main.evaluate(args)
