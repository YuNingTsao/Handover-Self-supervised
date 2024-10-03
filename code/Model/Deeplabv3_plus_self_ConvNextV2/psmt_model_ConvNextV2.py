import torch.nn
import torch.nn.functional as F

import numpy as np

from Model.Deeplabv3_plus.encoder_decoder import *
from Utils.losses import *
from Utils.self_loss import *



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

        self.encoder = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                       pretrained_model=None, back_bone=config['resnet'])
        self.decoder = DecoderNetwork(num_classes=num_classes, data_shape=config['data_h_w'])

        self.dice_loss = DiceLoss(num_classes)
        

    def warm_up_forward(self, x):
        f, mask = self.encoder(x)
        loss, output_ul, _ = self.decoder(f, mask)
        outputs = {'self_pred': output_ul}
        return loss, outputs
    
    def forward(self, input_ul=None, target_ul=None, x_FA=None, x_ICG=None, target_l=None,
                warm_up=False, mix_up=False):
        if mix_up:
            for i in range(0, int(x_FA.shape[0])): 
                x_FA[i], target_l[i] = mixup_data_FAICG(input_fa = x_FA[i], target_fa = target_l[i],
                                                        input_icg = x_ICG[i], target_icg = target_l[i])

        if warm_up:
            return self.warm_up_forward(x=input_ul, y=target_ul)
        else:
            f = self.encoder(input_ul)
            loss, output = self.decoder(f,data_shape=[input_ul.shape[-2], input_ul.shape[-1]])
            return loss, output
    
class Student_Net(nn.Module):
    def __init__(self, num_classes, config, sup_loss=None, 
                 cons_w_unsup=None, pretrained=True, weakly_loss_w=0.4):
        super(Student_Net, self).__init__()

        self.encoder = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2d, pretrained_model=res_net_2.format(str(config['resnet'])),back_bone=config['resnet'])
        self.decoder = VATDecoderNetwork(num_classes=num_classes, data_shape=config['data_h_w'])

        self.unsup_loss_w = cons_w_unsup
        self.unsuper_loss = semi_ce_loss
        self.dice_loss = DiceLoss(num_classes)
    
    def warm_up_forward(self, x, y):
        f = self.encoder(x)
        output_ul = self.decoder(fa = f)

        loss = self.masked_loss(torch.sigmoid(output_ul), torch.sigmoid(y[:]).long())
        outputs = {'self_pred': output_ul}
        return loss, outputs
    
    def forward(self, x_FA=None, x_ICG=None, target_l=None, x_ul=None, target_ul=None,
                warm_up=False, mix_up=False, semi_p_th=0.6, semi_n_th=0.0, 
                epoch=None, curr_iter = None, t1=None, t2=None):
        if warm_up:
            return self.warm_up_forward(x=x_FA, y=target_l)

        # labeled data prediction
        x_l = x_FA
        f = self.encoder(x_l)
        output_l = self.decoder(fa=f, icg=f, t_model=[t1.decoder, t2.decoder])
        # supervised loss
        loss_sup = self.dice_loss(torch.sigmoid(output_l), torch.sigmoid(target_l[:]).long())
        curr_losses = {'loss_sup': loss_sup}
        
        #unlabelled data prediction
        labels1, labels2, lamb = None, None, None
        if mix_up: 
            imgs, labels1, labels2, lamb = mixup_data(x_ul, target_ul)
            f = self.encoder(imgs)
            output_ul = self.decoder(fa=f, icg=f, t_model=[t1.decoder, t2.decoder])
            loss_unsup, pass_rate, neg_loss = self.unsuper_loss(inputs=output_ul, targets=target_ul,
                                                                label_1 = labels1, label_2 = labels2, lamb = lamb,
                                                                conf_mask=True, mix_up=mix_up,
                                                                threshold=semi_p_th, threshold_neg=semi_n_th)
        else:
            f = self.encoder(x_ul)
            output_ul = self.decoder(fa=f, icg=f, t_model=[t1.decoder, t2.decoder])
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
