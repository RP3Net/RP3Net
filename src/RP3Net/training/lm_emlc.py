import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.autograd import forward_ad
import lightning.pytorch.cli as L_cli
import lightning.pytorch.utilities as L_util

from . import lm
from .. import util
from .. import model

log = util.get_logger(__name__)
# log.setLevel(logging.DEBUG)


class EmlcLM(lm.RP3LM):
    """
    Need to run this with `ddp_find_unused_parameters_true: true` in the trainer config. 
    `static_graph = True` will not work (https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
    """

    def __init__(self, hypers:dict) -> None:
        super().__init__(hypers)
        self.student = self.model
        self.teacher = model.load_model(self.hypers.model)
        self.enhancer = TeacherEnhancer(2, **self.hypers.enhancer)
        self.automatic_optimization = False
        self.submodel_keys_list = ['student', 'teacher', 'enhancer']
        self.submodel_keys_map = {k: i for i, k in enumerate(self.submodel_keys_list)}
        self.emlc_k = self.hypers.emlc_k
        self.eta = float('nan')
        self.clean_sources = [self.sources_map[s] for s in self.hypers.clean_sources]

    def setup(self, stage):
        super().setup(stage)

    def build_loss(self):
        return None

    def student_opt(self):
        return self.optimizers()[self.submodel_keys_map['student']]
    
    def teacher_opt(self):
        return self.optimizers()[self.submodel_keys_map['teacher']]
    
    def enhancer_opt(self):
        return self.optimizers()[self.submodel_keys_map['enhancer']]
    
    def configure_optimizers(self) -> L_util.types.OptimizerLRScheduler:
        student_opt = L_cli.instantiate_class(self.student.parameters(), self.hypers.opt['student'])
        self.eta = self.hypers.opt['student']['init_args']['lr']
        teacher_opt = L_cli.instantiate_class(self.teacher.parameters(), self.hypers.opt['teacher'])
        enhancer_opt = L_cli.instantiate_class(self.enhancer.parameters(), self.hypers.opt['enhancer'])
        student_lrs = L_cli.instantiate_class(student_opt, self.hypers.lrs['student'])
        teacher_lrs = L_cli.instantiate_class(teacher_opt, self.hypers.lrs['teacher'])
        enhancer_lrs = L_cli.instantiate_class(enhancer_opt, self.hypers.lrs['enhancer'])
        student_lrs_conf = {'scheduler': student_lrs} | self.hypers.lrs_conf.student.to_dict()
        teacher_lrs_conf = {'scheduler': teacher_lrs} | self.hypers.lrs_conf.teacher.to_dict()
        enhancer_lrs_conf = {'scheduler': enhancer_lrs} | self.hypers.lrs_conf.enhancer.to_dict()
        self.lrs = [
            (student_lrs, student_lrs_conf),
            (teacher_lrs, teacher_lrs_conf),
            (enhancer_lrs, enhancer_lrs_conf),
        ]
        self.student_lrs = student_lrs
        return [student_opt, teacher_opt, enhancer_opt], [student_lrs_conf, teacher_lrs_conf, enhancer_lrs_conf]
    
    def training_step(self, batch, batch_idx):
        clean_batch, noisy_batch = util.torch_split_key_index(batch, 'source', torch.tensor(self.clean_sources, device=batch['source'].device))
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Step: {self.trainer.global_step}; Max sequence length: {clean_batch['seq']['input_ids'].shape[1]};")
            log.debug(f"Clean batch dataloader csv index: {clean_batch['idx'].tolist()};")
            log.debug(f"Noisy batch dataloader csv index: {noisy_batch['idx'].tolist()};")
        loss, loss_clean, loss_noisy = self.emlc(self.emlc_k, self.student, self.student_opt(),
                                            self.teacher,
                                            self.enhancer,
                                            noisy_batch, noisy_batch['yield_binary'],
                                            clean_batch, clean_batch['yield_binary'],
                                            self.eta,
                                            mix_g=self.hypers.mix_g,
                                            auxiliary_loss=self.hypers.auxiliary_loss,
                                            jvp_ad_method=self.hypers.jvp_ad_method)   
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Loss: {loss}, loss_clean: {loss_clean}, loss_noisy: {loss_noisy}")     
        self.teacher_opt().zero_grad()
        self.enhancer_opt().zero_grad()
        self.manual_backward(loss)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Teacher grad = {self.teacher.cls_head.layers[0].layers[0].weight.grad.abs().sum().item()}")
            log.debug(f"Enhancer grad = {self.enhancer.label_embedder.weight.grad.abs().sum().item()}")
        self.teacher_opt().step()
        self.enhancer_opt().step()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True, batch_size=batch['yield_binary'].shape[0])
        

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        for lrs, lrs_conf in self.lrs:
            lrs.step(self.trainer.callback_metrics[lrs_conf['monitor']])
        self.eta = self.student_lrs.get_last_lr()[0]

    def force_train_on_fit_start(self) -> None:
        """
        I think the PL logic that the parent method works around does not apply for manual optimization. 
        And we do not want to have different values of the training flag for the student and the teacher.
        """
        pass

    
    ########################### MULTI STEP META #################################
    # teacher_backward_ms
    def emlc(self, k, student, student_opt,
            teacher,
            enhancer,
            data_noisy, label_noisy:torch.Tensor,
            data_clean, label_clean:torch.Tensor,
            eta,
            num_classes=2,
            injection_mode='adversarial',
            jvp_ad_method='forward',
            mix_g=True,
            auxiliary_loss=False
            ):
        # Teacher forward pass
        bs_noisy, bs_clean = label_noisy.shape[0], label_clean.shape[0]
        if auxiliary_loss:
            data_all = util.torch_concat_chunks([data_noisy , data_clean])
            teacher_logits_all, teacher_repr_all = teacher(data_all, return_repr=True)
            teacher_logits_noisy, teacher_logits_clean = teacher_logits_all.split((bs_noisy, bs_clean))
            teacher_repr_noisy, teacher_repr_clean = teacher_repr_all.split((bs_noisy, bs_clean))
        else:
            teacher_logits_noisy, teacher_repr_noisy = teacher(data_noisy, return_repr=True)
        if mix_g:
            data_clean_student, data_clean_teacher = util.torch_split_chunks(data_clean, 2)
            label_clean_student, label_clean_teacher = util.torch_split_chunks(label_clean, 2) 
        
        # given current meta net, get corrected label
        label_corrected = self.enhance_labels(teacher_logits_noisy, teacher_repr_noisy, label_noisy, enhancer, num_classes)

        old_nets = []
        data_noisy_chunks = util.torch_split_chunks(data_noisy, k)
        label_corrected_chunks = util.torch_split_chunks(label_corrected, k)
        loss_noisy = torch.zeros(1, device=teacher_logits_noisy.device)
        
        for step in range(k):
            # Fetch step data
            data_step = data_noisy_chunks[step]
            label_step = label_corrected_chunks[step].detach()
            # Update student
            if mix_g:
                old_main_net, loss = self.update_student(student, student_opt, data_step, label_step, 
                                                         data_clean_student, label_clean_student)
            else:
                old_main_net, loss = self.update_student(student, student_opt, data_step, label_step)
            old_nets.append(old_main_net)
            loss_noisy += loss
        loss_noisy /= k

        # compute gw for updating meta_net
        if mix_g:
            gw, loss_clean = self.compute_gw(student, data_clean_teacher, label_clean_teacher)
        else:
            gw, loss_clean = self.compute_gw(student, data_clean, label_clean)

        # Gather all jvps
        gamma = 1 - eta
        discount_factor = 1
        discounted_jvps = []
        for step in reversed(range(k)):

            # Fetch step data
            data_step = data_noisy_chunks[step]
            model_step = old_nets[step]

            jvp = self.jacobian_vector_product(model_step, data_step, gw, jvp_ad_method) # Compute jacobian vector product
            discounted_jvps.append(jvp.data * discount_factor)

            discount_factor *= gamma # Update the discount factor
        
        # Meta loss
        discounted_jvps = torch.cat(discounted_jvps)
        loss = self.meta_loss(discounted_jvps, label_corrected, eta)
        
        if auxiliary_loss:
            # Supervised loss
            sup_loss = F.cross_entropy(teacher_logits_clean, label_clean)

            # Label retaining binary loss
            retain_loss = self.retain_conf_loss(enhancer, teacher_repr_clean, teacher_logits_clean,
                                            label_clean, num_classes,
                                            mode=injection_mode)

            # Update meta net
            loss += (sup_loss + retain_loss)

        return loss, loss_clean, loss_noisy

    ########################### SINGLE STEP META #################################

    # def teacher_backward(self, student, student_opt,
    #                     teacher,
    #                     enhancer,
    #                     data_noisy, label_noisy, 
    #                     data_clean, label_clean,
    #                     eta, 
    #                     num_classes=2,
    #                     injection_mode='adversarial', 
    #                     jvp_ad_method='forward',
    #                     mix_g=True
    #                     ):
        
    #     # Teacher forward pass
    #     bs_s, bs_g = data_noisy.shape[0], data_clean.shape[0]
    #     all_data = torch.cat([data_noisy , data_clean])
    #     teacher_logit_all, teacher_repr_all = teacher(all_data, return_h=True)
    #     teacher_logits_noisy, teacher_logits_clean = teacher_logit_all.split((bs_s, bs_g))
    #     teacher_repr_noisy, teacher_repr_clean = teacher_repr_all.split((bs_s, bs_g))
    #     if mix_g:
    #         data_clean_student, data_clean_teacher = data_clean.chunk(2)
    #         label_clean_student, label_clean_teacher = label_clean.chunk(2) 
        
    #     # given current meta net, get corrected label
    #     pseudo_label_noisy = self.enhance_labels(teacher_logits_noisy, teacher_repr_noisy, label_noisy, enhancer, num_classes)

    #     # Update student
    #     # Compute clean grad w.r.t student eval data
    #     if mix_g:
    #         old_student, loss_noisy = self.update_student(student, student_opt,
    #                                                 data_noisy, pseudo_label_noisy.detach(),
    #                                                 data_clean_student, label_clean_student)
    #         gw, loss_clean = self.compute_gw(student, data_clean_teacher, label_clean_teacher)
    #     else:
    #         old_student, loss_noisy = self.update_student(student, student_opt,
    #                                                 data_noisy, pseudo_label_noisy.detach())
    #         gw, loss_clean = self.compute_gw(student, data_clean, label_clean)

    #     # Compute jacobian vector product
    #     jvp = self.jacobian_vector_product(old_student, data_noisy, gw, jvp_ad_method)

    #     # Meta loss
    #     reg_loss = self.meta_loss(jvp, pseudo_label_noisy, eta)
        
    #     # Supervised loss
    #     sup_loss = F.cross_entropy(teacher_logits_clean, label_clean)

    #     # Label retaining binary loss
    #     retain_loss = self.retain_conf_loss(enhancer, teacher_repr_clean, teacher_logits_clean,
    #                                     label_clean, num_classes, mode=injection_mode)

    #     # Update meta net
    #     # t_loss = reg_loss
    #     loss = sup_loss + retain_loss + reg_loss
    #     return loss, loss_clean, loss_noisy


    ########################### Extra Utils #################################

    def update_student(self, student, student_opt, data_noisy, pseudo_label_noisy, data_clean_student=None, label_clean_student=None):
        # a copy of the old student
        old_main_net = copy.deepcopy(student)

        if data_clean_student is not None:
            # compute logits
            bs_noisy, bs_clean = pseudo_label_noisy.shape[0], label_clean_student.shape[0]
            all_data = util.torch_concat_chunks([data_noisy, data_clean_student])
            all_logits = student(all_data)
            logits_s, logits_g = all_logits.split((bs_noisy, bs_clean))
            
            # compute loss for updating the student
            loss_s = F.cross_entropy(logits_s, pseudo_label_noisy)
            loss_g = F.cross_entropy(logits_g, label_clean_student)
            loss = loss_s + loss_g
        else:
            logits = student(data_noisy)
            loss = F.cross_entropy(logits, pseudo_label_noisy)

        # Update main nets
        student_opt.zero_grad()
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Student layer = {self.student.fm.base_model.model.encoder.layer[-1].attention.self.query.lora_A.default.weight.flatten()[:10]}")
        self.manual_backward(loss)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Student grad = {self.student.fm.base_model.model.encoder.layer[-1].attention.self.query.lora_A.default.weight.grad.flatten()[:10]}")
        student_opt.step()
        return old_main_net, loss

    def enhance_labels(self, teacher_logits_noisy, teacher_repr_noisy, label_noisy, enhancer, num_classes):
        retain_conf = enhancer(teacher_repr_noisy, label_noisy)
        preds = F.softmax(teacher_logits_noisy, dim=1)
        one_hot = F.one_hot(label_noisy, num_classes)
        pseudo_label_s = retain_conf * one_hot + (1-retain_conf) * preds
        return pseudo_label_s

    def compute_gw(self, student:nn.Module, data_clean:torch.Tensor, label_clean:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # compute gw for updating meta_net
        logit_clean = student(data_clean)
        loss_clean = F.cross_entropy(logit_clean, label_clean)
        student.zero_grad()
        self.manual_backward(loss_clean)
        gw = [param.grad.data for param in student.parameters() if param.requires_grad]
        # DONT DO OPTIMIZATION STEP
        return gw, loss_clean

    def meta_loss(self, jvp, pseudo_label, eta):
        # Meta loss
        batch_dot_product = (jvp.data * pseudo_label).sum(1)
        meta_loss = eta * batch_dot_product.mean() # Batch dot product
        return meta_loss

    def retain_conf_loss(self, enhancer, teacher_repr_clean, teacher_logit_clean, label_clean, num_classes, mode='adversarial'):
        # Label retaining binary loss
        device = teacher_repr_clean.device
        if mode == 'random':
            label_clean_fake = torch.clone(label_clean)
            label_clean_fake[::2] = torch.randint_like(label_clean_fake[::2], high=num_classes)
        elif mode == 'adversarial':
            top_two_preds = torch.topk(teacher_logit_clean, 2, dim=1, sorted=True)[1]
            adversarial_labels = torch.where(top_two_preds[:,0] != label_clean, top_two_preds[:,0], top_two_preds[:,1])
            label_clean_fake = torch.clone(label_clean)
            label_clean_fake[::2] = adversarial_labels[::2].to(device)
        else:
            return 0
        label_clean_mask = torch.eq(label_clean_fake, label_clean).type(torch.float).to(device)
        retain_conf_clean = enhancer(teacher_repr_clean, label_clean_fake)
        retain_conf_clean = torch.clamp(retain_conf_clean, min=0, max=1) # prevents floating point errors
        retain_conf_loss = F.binary_cross_entropy(retain_conf_clean, label_clean_mask.reshape_as(retain_conf_clean))
        return retain_conf_loss

    def jacobian_vector_product(self, model, inputs, vector, method='forward'):
        if method == 'forward':
            '''
            jvp products using forward mode AD as demonstrated in:
            https://pytorch.org/tutorials/intermediate/forward_ad_usage.html
            '''
            with torch.no_grad():
                with forward_ad.dual_level():
                    params = {}
                    if type(vector) is torch.Tensor:
                        offset = 0
                    i = 0
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            if type(vector) is torch.Tensor:
                                vector_part = vector[offset: offset+p.nelement()].view(p.size())
                                offset += p.nelement()
                                params[name] = forward_ad.make_dual(p, vector_part)
                            else:
                                params[name] = forward_ad.make_dual(p, vector[i])
                                i += 1


                    out = functional_call(model, params, inputs)
                    lsm = F.log_softmax(out, dim=1)
                    jvp = torch.autograd.forward_ad.unpack_dual(lsm).tangent
            return jvp
        elif method == 'double-back-trick':
            '''
            jvp products using double backward as demonstrated in:
            https://j-towns.github.io/2017/06/12/A-new-trick.html
            https://discuss.pytorch.org/t/forward-mode-ad-vs-double-grad-trick-for-jacobian-vector-product/159037
            '''
            out = model(inputs)
            lsm = F.log_softmax(out, dim=1)
            v = torch.zeros_like(lsm, requires_grad=True)
            params = [p for p in model.parameters() if p.requires_grad]
            g = torch.autograd.grad(lsm, params, v, create_graph=True)
            jvp = torch.autograd.grad(g, v, vector)[0]
            return jvp.detach()
        else:
            raise NotImplementedError

    
class TeacherEnhancer(nn.Module): # EMLC
    def __init__(self, num_classes, embedding_dim, label_embedding_dim, hidden_dim):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.label_embedder = nn.Embedding(self.num_classes, label_embedding_dim)

        in_dim = embedding_dim + label_embedding_dim

        self.retain_label = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, y_n):
        y_emb = self.label_embedder(y_n)
        hin = torch.cat([features, y_emb], dim=-1)
        retain_logit = self.retain_label(hin)
        retain_conf = torch.sigmoid(retain_logit)
        return retain_conf
    

