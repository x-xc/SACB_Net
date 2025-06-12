from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from dataset import datasets, trans
import numpy as np
import torch
import random
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from model import SACB_Net
import csv

os.environ["base_dir"] = '/bask/projects/d/duanj-ai-imaging/xxc/dataset_all'

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def setup_seed(seed, cuda_deterministic=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    g = torch.Generator()
    g.manual_seed(0)
    setup_seed(seed=0, cuda_deterministic=False)
    k = 9 
    dataset=['ixi', 'lpba', 'abd'][2]
    tag = f'{dataset}_k{k}'
    bs = 1
    weights = [1, 0.3] # loss weights [1, 3], [1, 1], [1, 0.3-0.5]
    
    save_dir  = 'sacb_ncc_{}_reg_{}_{}/'.format(weights[0], weights[1], tag)
   
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    # sys.stdout = Logger('logs/'+save_dir)
    os.makedirs('./csv', exist_ok=True)
    csv_name  = './csv/sacb_ncc_{}_reg_{}_{}.csv'.format(weights[0], weights[1], tag)
    
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        csvwriter = csv.DictWriter(f, fieldnames=fnames)
        csvwriter.writeheader()
        
    lr = 1e-4 # learning rate
    epoch_start = 0
    max_epoch = 300 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize dataset
    '''
    if dataset == 'ixi':
        atlas_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/atlas.pkl')
        train_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/Train/')
        val_dir   = os.path.join(os.getenv('base_dir'), 'IXI_data/Val/')
        train_composed  = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed    = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label
                                            trans.NumpyType((np.float32, np.int16))])
        train_set  = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
        val_set    = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
        dice_score = utils.dice_val_VOI
        img_size   = (160,192,224)
    if dataset == 'lpba':
        train_dir = os.path.join(os.getenv('base_dir'), 'LPBA_data_2/Train/')
        val_dir   = os.path.join(os.getenv('base_dir'), 'LPBA_data_2/Val/')
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed   = transforms.Compose([trans.Seg_norm2(), #rearrange segmentation label
                                            trans.NumpyType((np.float32, np.int16))])
        train_set  = datasets.LPBABrainDatasetS2S(sorted(glob.glob(train_dir + '*.pkl')),      transforms=train_composed)
        val_set    = datasets.LPBABrainInferDatasetS2S(sorted(glob.glob(val_dir + '*.pkl')),   transforms=val_composed)
        dice_score = utils.dice_LPBA
        img_size   = (160, 192, 160)
    if dataset == 'abd':
        train_dir = os.path.join(os.getenv('base_dir'), 'AbdomenCTCT/Train/')
        val_dir   = os.path.join(os.getenv('base_dir'), 'AbdomenCTCT/Val/')
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed   = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        train_set      = datasets.LPBABrainDatasetS2S(sorted(glob.glob(train_dir + '*.pkl')),      transforms=train_composed)
        val_set    = datasets.LPBABrainInferDatasetS2S(sorted(glob.glob(val_dir + '*.pkl')),   transforms=val_composed)
        dice_score = utils.dice_abdo
        img_size   = (192, 160, 224)
    
    train_loader = DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)


    '''
    Initialize model
    '''
    model =  SACB_Net(inshape=img_size,num_k=k)
    model.set_k(k)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.SpatialTransformer(size=img_size, mode='nearest').cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''

    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num of params:', params)
    
    optimizer = torch.optim.Adam(model.parameters(), lr= updated_lr)
    
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            output = model(x, y)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write("\r" + 'Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))
            sys.stdout.flush()
            
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
             
                _,flow = model(x, y)
                def_out = reg_model(x_seg.float(), flow)
               
                dsc = dice_score(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
             
        with  open(csv_name, 'a') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([epoch, eval_dsc.avg])
            
        best_dsc = max(eval_dsc.avg, best_dsc) 
        save_checkpoint({
            # 'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            # 'best_dsc': best_dsc,
            # 'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.4f}_e{}.pth.tar'.format(eval_dsc.avg,epoch))
        
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
    writer.close()

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power), 8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=20):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
