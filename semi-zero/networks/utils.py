import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.autograd.variable as Variable

def perb_att(input_att):
    gama = torch.rand(input_att.size())
    noise = gama * input_att
    att = norm(1)(noise+input_att)
    return att

class norm(nn.Module):
    def __init__(self,radius=1):
        super().__init__()
        self.radius = radius
    def forward(self,x):
        x = self.radius * x/ torch.norm(x,p=2,dim=-1,keepdim=True)
        return x  

def reconstruct_W1_loss(h_att,att):
    wt = (h_att-att).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (h_att-att).abs()
    return loss.sum()/loss.size(0)



def loss_fn(opt, recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)

loss_2= torch.nn.MSELoss()
def loss_fn_2(opt, recon_x, x, mean, log_var):
    mse = loss_2(recon_x,x.detach())
    mse = mse.sum()/x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    loss_all = mse + KLD 
    return loss_all



def generate_syn_feature(opt,netG, classes, attribute, num, return_norm = False):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.attSize)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            batch_att = iclass_att.repeat(num, 1)
            if opt.perb:
                batch_att = perb_att(batch_att)
            syn_att.copy_(batch_att)
            syn_noise.normal_(0, opt.tr_sigma)
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)
    if return_norm:
        out_notNorm = netG.get_out()
        return syn_feature, syn_label , out_notNorm
    else:
        return syn_feature, syn_label

def calc_gradient_penalty(opt, netD, real_data, fake_data, input_att = None,lambda1 = 1):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # L_2 norm for iterpolated data
    if opt.L2_norm:
        interpolates = opt.radius * interpolates / torch.norm(interpolates,p=2,dim=1,keepdim=True)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    if input_att is not None:
        disc_interpolates = netD(interpolates, Variable(input_att))
    else:
        disc_interpolates = netD(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty


def calc_ood_loss(feature, label, map_label, opt, data, netG, netMap, netF_CLS, netOODMap, netOOD, net_sigmoid, final_criterion, KL_Loss, bi_criterion):
    #
    syn_feature, syn_label = generate_test(netG, data.unseenclasses, data.attribute, opt.ood_train_syn_num,opt)
    syn_map_label = map_label_all(syn_label, data.seenclasses, data.unseenclasses, data.ntrain_class)
    netF_CLS.zero_grad()
    netMap.zero_grad()
    net_sigmoid.zero_grad()

    if opt.cuda:
        syn_feature = syn_feature.cuda()
        syn_label = syn_label.cuda()
        syn_map_label = syn_map_label.cuda()
    start_select = syn_feature.size(0)
    syn_feature = torch.cat((syn_feature, feature), 0)
    syn_label = torch.cat((syn_label, label), 0)
    syn_map_label = torch.cat((syn_map_label, map_label), 0)
    end_select = syn_feature.size(0)

    #
    embed = netMap(syn_feature)
    output, bi_feature, kl_input = netF_CLS(embed, data.seenclasses, data.unseenclasses)
    final_cls_loss = final_criterion(output, syn_label)

    # ood
    embed_O = netOODMap(syn_feature)
    ood_output, logit_v = netOOD(embed_O)
    energy_score, _ = torch.max(ood_output, dim=1, keepdim=True)

    # kl loss
    #
    indices = torch.arange(start_select, end_select).cuda()

    # softmax kl loss
    seen_kl_input = torch.index_select(kl_input, dim=0, index=indices)
    seen_logit_v = torch.index_select(logit_v, dim=0, index=indices)
    logits_distillation_loss = KL_Loss(F.log_softmax(seen_kl_input, dim=1), F.softmax(seen_logit_v, dim=1))

    # embed distillation loss
    embed_teacher = torch.index_select(embed_O, dim=0, index=indices)
    embed_student = torch.index_select(embed, dim=0, index=indices)
    # embed_distillation_loss = lrd_loss(embed_student, embed_teacher)

    label_embed_teacher = torch.index_select(syn_label, dim=0, index=indices)
    label_embed_student = torch.index_select(syn_label, dim=0, index=indices)
    embed_distillation_loss = batch_embed_distillation(embed_teacher, embed_student, label_embed_teacher,
                                                       label_embed_student, bi_criterion)

    # kl_loss = KL_Loss(F.log_softmax(kl_input, dim=1), F.softmax(logit_v, dim=1))
    # sigmoid
    Sequence_label = net_sigmoid(energy_score)
    OOD_confidence = 1 - Sequence_label
    Sequence_label = torch.cat((Sequence_label, OOD_confidence), 1)
    OOD_contrastive_loss = batch_cosine_similarity(F.softmax(bi_feature, dim=1), F.softmax(Sequence_label, dim=1),
                                                   syn_label, bi_criterion)
    ############################################
    # OOD_distillation_loss = fig_OOD_distillation_loss(bi_feature, Sequence_label)

    ############################################
    OOD_loss = OOD_contrastive_loss
    # OOD_loss = OOD_contrastive_loss + OOD_distillation_loss
    # distillation_loss = logits_distillation_loss
    distillation_loss = embed_distillation_loss + logits_distillation_loss
    return final_cls_loss, OOD_loss, distillation_loss


def generate_test(netG, classes, attribute, num, opt):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor()
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attSize)
    syn_noise = torch.FloatTensor(nclass * num, opt.attSize)
    # syn_feature = Variable(syn_feature, requires_grad=True)

    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        # syn_feature = syn_feature.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_noise.normal_(0, 1)

    output = netG(syn_noise, syn_att).cpu()
    # syn_feature = torch.cat((syn_feature, output), 1)

    return output, syn_label


def map_label_all(label, seenclasses, unseenclasses, _nclass_s):
    mapped_label = torch.LongTensor(label.size())
    nclass_s = _nclass_s
    for i in range(seenclasses.size(0)):
        mapped_label[label == seenclasses[i]] = i

    for j in range(unseenclasses.size(0)):
        mapped_label[label == unseenclasses[j]] = j + nclass_s

    return mapped_label


def batch_embed_distillation(real_seen_feat, syn_seen_feat, real_seen_label, syn_seen_label, bi_criterion):
    a_number = real_seen_feat.size(0)
    b_number = syn_seen_feat.size(0)
    a_embedding = real_seen_feat.unsqueeze(1).repeat(1, b_number, 1).view(-1, real_seen_feat.size(1))
    b_embedding = syn_seen_feat.unsqueeze(0).repeat(a_number, 1, 1).view(-1, syn_seen_feat.size(1))

    similarity = (torch.cosine_similarity(a_embedding, b_embedding) + 1) / 2
    similarity = similarity.view(similarity.size(0), -1)

    real_seen_label = real_seen_label.contiguous().view(1, -1)
    syn_seen_label = syn_seen_label.contiguous().view(-1, 1)

    # 计算ground_truth_label
    ground_truth_label = torch.eq(real_seen_label, syn_seen_label).float().view(-1, 1)

    batch_embed_loss = bi_criterion(similarity, ground_truth_label)

    return batch_embed_loss


def batch_cosine_similarity(a,b,syn_label,bi_criterion):
    a_number = a.size(0)
    b_number = b.size(0)
    a_embedding = a.unsqueeze(1).repeat(1, b_number, 1).view(-1, a.size(1))
    b_embedding = b.unsqueeze(0).repeat(a_number, 1, 1).view(-1, b.size(1))
     # Compute cosine similarity and rescale it to the range [0, 1]
    similarity = (torch.cosine_similarity(a_embedding, b_embedding) + 1) / 2
    # similarity = torch.cosine_similarity(a_embedding, b_embedding)
    similarity = similarity.view(similarity.size(0), -1)
    syn_label = syn_label.contiguous().view(-1, 1)
    ground_truth_label = torch.eq(syn_label, syn_label.T).float().view(-1, 1)

    # Apply logit transformation to similarity
    # logits = torch.log(similarity / (1 - similarity))
    # OOD_contrastive_loss = bi_criterion(logits, ground_truth_label)
    OOD_contrastive_loss = bi_criterion(similarity, ground_truth_label)

    return OOD_contrastive_loss