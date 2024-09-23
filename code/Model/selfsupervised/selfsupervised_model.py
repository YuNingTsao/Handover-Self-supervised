import torch.nn
import torch.nn.functional as F

import numpy as np

from Model.selfsupervised.selfsupervised_encoder_decoder import *
from Utils.losses import *



#mix FA & ICG data with given portion as input data
# used by teacher
def mixup_data_FAICG(input_fa, target_fa, input_icg, target_icg, alpha=1):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    mixed_input = lamb * input_fa + (1 - lamb) * input_icg
    mixed_target = lamb * target_fa + (1 - lamb) * target_icg

    return mixed_input, mixed_target

# used by student
def mixup_data(inputs, labels, alpha = 1, use_cuda = True):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1
    batch_size = inputs.size(0)
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
        
    mixed_inputs = lamb * inputs + (1 - lamb) * inputs[index, :]
    label_a, label_b = labels, labels[index]
    
    return mixed_inputs, label_a, label_b, lamb

class Teacher_Net(nn.Module):
    def __init__(self, num_classes, config, sup_loss=None, 
                 cons_w_unsup=None, pretrained=True, weakly_loss_w=0.4):
        super(Teacher_Net, self).__init__()

#       self.maskAutoEncoder = nn.AutoEncoder
        self.maskAutoEncoder = MaskedAutoencoder()        
        self.segmentationMAE = MaskedAutoencoderForSegmentation(pretrained_encoder=self.maskAutoEncoder)

    def warm_up_forward(self, input=None):
        loss, output = self.maskAutoEncoder(input)
        outputs = {'self_pred': output}
        return loss, outputs
   
    #    
    def forward(self, input_ul=None, target_ul=None, x_FA=None, x_ICG=None, target_l=None, warm_up=False, mix_up=False):
        if mix_up:
            for i in range(0, int(x_FA.shape[0])): 
                x_FA[i], target_l[i] = mixup_data_FAICG(input_fa = x_FA[i], target_fa = target_l[i],
                                                        input_icg = x_ICG[i], target_icg = target_l[i])
        if warm_up: #warm up without segmentation groundtruth (ordinary MAE)
            return self.warm_up_forward(input_ul)
            
        else: #unlabeled prediction in semi-train
            loss, output = self.segmentationMAE(input_ul, target_ul)
            return loss, output
    
class Student_Net(nn.Module):
    def __init__(self, num_classes, config, sup_loss=None, 
                 cons_w_unsup=None, pretrained=True, weakly_loss_w=0.4):
        super(Student_Net, self).__init__()

        self.maskAutoEncoder = MaskedAutoencoder()
        self.segmentationMAE = MaskedAutoencoderForSegmentation(pretrained_encoder=self.maskAutoEncoder)
        self.unsup_loss_w = cons_w_unsup
        self.unsuper_loss = semi_ce_loss
        self.dice_loss = DiceLoss(num_classes)
        
    
    def warm_up_forward(self, input):
        loss, output = self.maskAutoEncoder(input)
        outputs = {'self_pred': output}
        return loss, outputs
    
    def forward(self, x_FA=None, x_ICG=None, target_l=None, x_ul=None, target_ul=None,
                warm_up=False, mix_up=False, semi_p_th=0.6, semi_n_th=0.0, 
                epoch=None, curr_iter = None, t1=None, t2=None):
        if warm_up: #warm up without segmentation groundtruth (ordinary MAE)
            return self.warm_up_forward(x_ul)

        # predict labeled data
        loss_sup, output_l = self.segmentationMAE(x_FA, target_l)
        # supervised loss
        curr_losses = {'loss_sup': loss_sup}
        
        # predict unlabeled data
        labels1, labels2, lamb = None, None, None
        if mix_up: 
            imgs, labels1, labels2, lamb = mixup_data(x_ul, target_ul)
            loss_unsup, output_ul = self.segmentationMAE(imgs, target_ul)
                                                                 
        else:
            loss_unsup, output_ul = self.segmentationMAE(x_ul, target_ul)
            # calculate consistency loss (loss of unalabled data prediction of teacher and student)
            loss_unsup, pass_rate, neg_loss = self.unsuper_loss(inputs=output_ul, targets=target_ul,
                                                                label_1 = labels1, label_2 = labels2, lamb = lamb,
                                                                conf_mask=True, mix_up=mix_up,
                                                                threshold=semi_p_th, threshold_neg=semi_n_th)

        if semi_n_th > .0:
            confident_reg = .5 * torch.mean(F.softmax(output_ul, dim=1) ** 2)
            loss_unsup += neg_loss
            loss_unsup += confident_reg

        loss_unsup = loss_unsup * self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
        total_loss = loss_unsup + loss_sup

        curr_losses['loss_unsup'] = loss_unsup
        curr_losses['pass_rate'] = pass_rate
        curr_losses['neg_loss'] = neg_loss
        outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}
        return total_loss, curr_losses, outputs
