import os
import random

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, AdamW
from transformers.activations import gelu
from transformers.file_utils import (add_code_sample_docstrings,
                                     add_start_docstrings,
                                     add_start_docstrings_to_model_forward,
                                     replace_return_docstrings)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import (BertLMPredictionHead,
                                                    BertModel,
                                                    BertPreTrainedModel)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead, RobertaModel, RobertaPreTrainedModel)

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None, 
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    do_mask=False,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)
    anc_input_ids = input_ids[:,0,:].clone().detach() #experimental stop-gradient
    anc_attention_mask = attention_mask[:,0,:]

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)) #[bsz/2, 1, nh] * [1, bsz/2, nh] -> [bsz/2, bsz/2]
    bml_loss = None

    #kmeans clustering
    if cls.model_args.kmeans > 0:
        normalized_cos = cos_sim * cls.model_args.temp
        avg_cos = normalized_cos.mean().item() 

        if not cls.cluster.initialized:
            if avg_cos <= cls.model_args.kmean_cosine:
                cls.cluster.optimized_centroid_init(z1, cos_sim*cls.model_args.temp)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("kmeans start!!")
        elif cls.cluster.initialized:
            _, dp_index, _ = cls.cluster._clustering(z1)
            cls.cluster(z1, normalized_cos)
            num_sent = 3 #to be fix
            z3, dp_hard, _ = cls.cluster.provide_hard_negative(z1)

        cls.cluster.global_step += 1


    # Hard negative
    if num_sent >= 3:
        z1_hard_negatives_cos = torch.zeros(dp_index.size(0), 64).to(cls.device)
        z1_false_negatives_cos = torch.zeros(dp_index.size(0), 64).to(cls.device)
        cls.cluster.add_cluster(z1, dp_index)
        for i in range(dp_index.size(0)):
            temp = cls.cluster.get_memory_bank(dp_hard[i]).to(cls.device)
            z1_hard_negatives_cos[i] = cls.sim(z1[i], temp.detach())

        #BLM loss is performed after the first eval step
        if cls.cluster.global_step >125:
            for i in range(dp_index.size(0)):
                temp2 = cls.cluster.get_memory_bank(dp_index[i]).to(cls.device)
                z1_false_negatives_cos[i] = cls.sim(z1[i], temp2.detach())
            bml_loss = cls.cluster.AdaptiveBfMLLoss(z1_false_negatives_cos.detach() * cls.model_args.temp,
                                              cls.sim(z1, cls.cluster.centroid[dp_index]).diagonal()*cls.model_args.temp,
                                              normalized_cos.diag())

        z1_hard_cen = cls.cluster.Gaussian_negative(z1, dp_hard) / cls.model_args.temp
        cos_sim_mind = torch.cat([z1_hard_cen, z1_hard_negatives_cos], 1)+cls.model_args.hard_negative_weight
        cos_sim = torch.cat([cos_sim, cos_sim_mind], 1)
        cos_sim = torch.where(cos_sim == 0.0, cos_sim - 1e12, cos_sim)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)
    if bml_loss is not None:
        loss = loss + cls.model_args.bml_weight * bml_loss


    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.dropout_prob is not None:
            config.attention_probs_dropout_prob = self.model_args.dropout_prob
            config.hidden_dropout_prob = self.model_args.dropout_prob           
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        if self.model_args.kmeans > 0:
            self.cluster = kmeans_cluster(config, self.model_args)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.dropout_prob is not None:
            config.attention_probs_dropout_prob = self.model_args.dropout_prob
            config.hidden_dropout_prob = self.model_args.dropout_prob           
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        if self.model_args.kmeans > 0:
            self.cluster = kmeans_cluster(config, self.model_args)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

class kmeans_cluster(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()
        self.model_args = model_args
        self.k = model_args.kmeans
        self.sim = Similarity(temp=1)
        self.initialized = False
        self.global_step = 0
        self.lr = model_args.kmeans_lr
        self.beta = model_args.bml_beta
        self.alpha = model_args.bml_alpha
        self.guss_weight = model_args.guss_weight
        self.bankfull = False
        self.clusters_sample = [torch.zeros((64, 768), device="cuda", dtype=torch.float16, requires_grad=False) for _ in range(self.k)]
        if model_args.kmean_debug:
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.writer = SummaryWriter("runs/kmeans-7798-nobml-%.1f" % (model_args.kmean_cosine))

    def provide_hard_negative(self, datapoints:torch.Tensor, intra_cos_sim:torch.Tensor=None):#将每个xi对应hard_negative质心的中hard negative样本、索引和相似度值
        D = self.centroid.data.shape[-1]
        if intra_cos_sim is None:
            intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        values, indices = torch.topk(intra_cos_sim, k=2, dim=-1)# 获取每个datapoint相似度最高的两个聚类中心的索引
        hard_neg_index = indices[:, 1].unsqueeze(-1).expand(-1, D)
        hard_negative = torch.gather(self.centroid.data, dim=0, index=hard_neg_index)
        return hard_negative.detach(), indices[:,1], values[:,1]

    def mask_false_negative(self,
                            datapoints:torch.Tensor, 
                            batch_cos_sim:torch.Tensor=None, #cos(xi, xj) [bsz, bsz]
    ):   
        if batch_cos_sim is None:
            batch_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        dp_centroid_cos, dp_index, _ = self._clustering(datapoints)
        dp_cluster, _ = self._intra_class_adjacency(dp_index) #(bsz, bsz)
        dp_centroid_cos = dp_centroid_cos.expand_as(dp_cluster) #(bsz, bsz)
        # false_negative_mask: {1:masked, 0:unmasked}
        # false_negative_mask = dp_cluster * (batch_cos_sim>dp_centroid_cos)
        false_negative_mask = dp_cluster 
        return false_negative_mask

    def _clustering(self, datapoints:torch.Tensor):
        '''
        find the cluster belong to and corresponding centroid for each datapoint

        return 
        dp_centroid_cos: [bsz, 1], indicating that cosine similarity between datapoints to centroid to which they belong
        dp_index: [bsz, 1], indicating that the indices of cluster to which datapoints belong
        intra_cos_sim: [bsz, bsz], indicating that cosine similarity between datapoints and centroids
        '''
        intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        dp_centroid_cos, dp_index = torch.max(intra_cos_sim, dim=-1, keepdim=True) #(bsz, 1)
        return dp_centroid_cos, dp_index, intra_cos_sim

    def _intra_class_adjacency(self, dp_index:torch.Tensor):
        r'''
        dp_index: indicating the indices of cluster to which datapoints belong

        return:
        dp_cluster: [bsz, bsz], indicating that which datapoints belong to same cluster
        index_dp: [k, bsz], indicating that which datapoints belong to the clusters
        '''

        B, device = dp_index.shape[0], dp_index.device
        onehot_index = F.one_hot(dp_index.squeeze(-1), self.k) #(bsz, k)
        index_dp = onehot_index.T #(k, bsz)
        
        #adjacency matrix that dp_cluster[i][j]==1 if xi and xj belong to identical cluster
        dp_cluster = torch.matmul(onehot_index.float(), index_dp.float()) 
        #set dp_cluster[i][i] = 0
        dp_cluster.fill_diagonal_(0) 
        return dp_cluster, index_dp

    def AdaptiveBMLLoss(self, c1, c2, c3):
        """
        c1: x_false negatives
        c2: x _hard negatives
        c3: positive negatives
        """
        c1 = torch.where(c1 >0.95,c1-1, c1)
        c2 = c2.unsqueeze(1)  # (batch_size, 1)
        c3 = c3.unsqueeze(1)  # (batch_size, 1)
        loss1 = F.relu(c1 + self.alpha - c3)  # (batch_size, n_fn)
        loss2 = F.relu(c2 + self.beta - c1)  # (batch_size, n_fn)
        loss = (loss1 + loss2).mean()

        return loss
    
    def forward(self, 
                datapoints:torch.Tensor, 
                batch_cos_sim:torch.Tensor, 
    ):
        B = datapoints.shape[0]
        device = datapoints.device
        datapoints = datapoints.clone().detach()
        intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        dp_index = torch.argmax(intra_cos_sim, dim=-1, keepdim=True) #(bsz, 1)
        dp_cluster, index_dp = self._intra_class_adjacency(dp_index)

        dp_centroid = torch.gather(self.centroid, dim=0, index=dp_index.expand_as(datapoints)) #set the centroid corresponding to the datapoint
        if self.model_args.kmean_debug:
            if not dist.is_initialized() or dist.get_rank() == 0:
                dp_hardneg, _, _ = self.provide_hard_negative(datapoints, intra_cos_sim)
                self.debug_stat(datapoints, dp_centroid, dp_hardneg, batch_cos_sim, dp_index.squeeze(-1), dp_cluster)

        self.update(datapoints, dp_centroid, index_dp)

    def debug_stat(
        self, 
        datapoints:torch.Tensor, 
        dp_centroid:torch.Tensor, 
        dp_hardneg:torch.Tensor,
        batch_cos_sim:torch.Tensor,
        dp_index:torch.Tensor,
        dp_cluster:torch.Tensor,
    ):
        with torch.no_grad():
            device = datapoints.device
            diag_mask = torch.diag(torch.ones(self.k)).to(device)
            diag_mask = 1 - diag_mask

            #Cosine Distance
            positive_avg_cos = batch_cos_sim.diag().mean()
            batch_avg_cos = batch_cos_sim.mean()
            centroid_cosine = self.sim(self.centroid.unsqueeze(1), self.centroid.unsqueeze(0))
            centroid_cosine *= diag_mask
            centroid_avg_cos = centroid_cosine.sum() / (self.k*(self.k-1))
            cent_dp_cosine = self.sim(datapoints.unsqueeze(1), dp_centroid.unsqueeze(0))
            cent_dp_max_cos = cent_dp_cosine.diag().max()
            cent_dp_min_cos = cent_dp_cosine.diag().min()
            cent_dp_avg_cos = cent_dp_cosine.diag().mean()
            hard_neg_cosine = self.sim(datapoints.unsqueeze(1), dp_hardneg.unsqueeze(0))
            hard_neg_max_cos = hard_neg_cosine.diag().max()
            hard_neg_min_cos = hard_neg_cosine.diag().min()
            hard_neg_avg_cos = hard_neg_cosine.diag().mean()

            member_cos_sim = batch_cos_sim * dp_cluster
            member_mask = torch.logical_not(dp_cluster).float()
            avg_member_cos = member_cos_sim.sum() / dp_cluster.count_nonzero()
            max_member_cos = member_cos_sim.max()
            min_member_cos = (member_cos_sim + member_mask).min()

            cosine_dict = {
                "inter-centroid": centroid_avg_cos, "intra-cos": cent_dp_avg_cos, "hn-cos": hard_neg_avg_cos,
                "positive-cos": positive_avg_cos, "batch-cos": batch_avg_cos, "member-cos": avg_member_cos
            }
            self.writer.add_scalars("cosine", cosine_dict, self.global_step)
            detail_cosine = {
                "max-intra-cos": cent_dp_max_cos, "min-intra-cos": cent_dp_min_cos,
                "max-hardneg-cos": hard_neg_max_cos, "min-hardneg-cos": hard_neg_min_cos,
                "max-member-cos": max_member_cos, "min-member-cos": min_member_cos
            }
            self.writer.add_scalars("detail_cosine", detail_cosine, self.global_step)

            #Euclidean Distance
            euc_distance = nn.PairwiseDistance()
            centroid_L2dist = torch.cdist(self.centroid, self.centroid)
            centroid_L2dist *= diag_mask
            centroid_avg_L2 = centroid_L2dist.sum() / (self.k*(self.k-1))
            cent_dp_L2dist = euc_distance(datapoints, dp_centroid)
            cent_dp_max_L2 = cent_dp_L2dist.max()
            L2dist_dict = {"inter-centroid": centroid_avg_L2, "max-intra-euc": cent_dp_max_L2}
            self.writer.add_scalars("euc_distance", L2dist_dict, self.global_step)

            #how many member in each cluster
            cluster_count = torch.zeros((self.k))

            for n in dp_index:
                try:
                    cluster_count[n] += 1
                except:
                    if dist.is_initialized() and dist.get_rank() == 0:
                        import pdb; pdb.set_trace()
            avg_cluster_member = cluster_count.sum() / cluster_count.count_nonzero()
            max_cluster_member = cluster_count.max()
            cluster_dict = {"avg_member": avg_cluster_member, "max_member": max_cluster_member}
            self.writer.add_scalars("cluster_member", cluster_dict, self.global_step)

    def optimized_centroid_init(self, centroid: torch.Tensor, batch_cos_sim: torch.Tensor):
        data = centroid.clone().detach()
        L = data.shape[0]
        print("k=" + str(self.k) + "  L=" + str(L))
        assert self.k <= L
        self.centroid = nn.Parameter(data=torch.zeros_like(data[:self.k]))
        idx = list(range(data.shape[0]))
        first_idx = random.randint(0, L - 1)
        self.centroid.data[0] = data[first_idx]
        self.enqueue(0, data[first_idx])
        last_idx = first_idx

        # heuristic initialize the centroid of Kmeans clustering
        for i in range(1, self.k):
            # set the last centroid in cos_sim to maxmimal, it will be ignored in the later centroid selection process.
            batch_cos_sim[:, last_idx] = 100
            next_idx = torch.argmin(batch_cos_sim[last_idx])
            self.centroid.data[i] = data[next_idx]
            self.enqueue(i, data[next_idx])
            last_idx = next_idx

        self.optimizer = AdamW(self.parameters(), lr=self.lr)  # used when self.optimization = adamw
        self.initialized = True



    def update(
        self,
        datapoints:torch.Tensor,
        dp_centroid:torch.Tensor=None,
        index_dp:torch.Tensor=None, #[k, bsz]
    ):
        data = torch.matmul(index_dp.float(), datapoints) / index_dp.sum(dim=1, keepdim=True).float().clamp(
            min=1e-10)
        data_num=index_dp.sum(dim=-1)
        if self.bankfull==True:
            denominator = data_num + self.clusters_sample[0].size(0)
            need_data = (data * data_num.unsqueeze(1) + self.centroid.data * self.clusters_sample[0].size(0)) / denominator.unsqueeze(1)
        else:
            sum = [item.sum(dim=1) for item in self.clusters_sample]
            if all( not (sum_item == 0).any().item() for sum_item in sum):
                self.bankfull==True
            non_zero_counts = [torch.count_nonzero(tensor.detach()).item() for tensor in sum]
            non_zero=torch.tensor(non_zero_counts).to(data_num.device)
            non_zero = torch.max(non_zero, torch.tensor(1).to(data_num.device))
            denominator = data_num +non_zero
            need_data = (data * data_num.unsqueeze(1) + self.centroid.data*non_zero.unsqueeze(1)  ) / denominator.unsqueeze(1)
            # for i in range(len(non_zero_counts)):
            #     if data_num[i]==0:
            #         continue
            #     elif non_zero_counts[i]==0 :
            #         need_data[i]=data[i]*data_num[i]/(data_num[i]+1)+self.centroid.data[i]/(data_num[i]+1)
            #     else:
            #         need_data[i]=data[i]*data_num[i]/(data_num[i]+non_zero_counts[i])+self.centroid.data[i]*non_zero_counts[i]/(data_num[i]+non_zero_counts[i])
        self.centroid.data = need_data

    def get_memory_bank(self, n):
        """
        Get all hard_negative samples for the specified centre of mass.
        """
        return self.clusters_sample[n]
    def enqueue(self, i, item):
        # Make sure i is a valid queue index
        # if not (0 <= i < len(self.clusters_sample[i])):
        #     raise IndexError("Queue index out of range")

        # Make sure item is a PyTorch tensor and is compatible with the queue data type and device
        if not isinstance(item, torch.Tensor):
            raise TypeError("Item must be a PyTorch tensor")
            #Adjust the shape of the item to match the shape of the queue (1, k)
        item = item.view(1, -1)

        # Get current queue
        queue = self.clusters_sample[i]

        # If the queue is full, remove the head element
        if queue.shape[0] == 64:
            queue = queue[1:]
        queue = queue
        item = item
        # Adding a new element to the end of the queue
        queue = torch.cat((queue, item), dim=0)

        # Updating the queue
        self.clusters_sample[i] = queue.to("cuda")

    def add_cluster(self,centroid:torch.Tensor,  dp_index:torch.Tensor):
        """
        Add a batch of samples to the centre-of-mass queue.
        centroid:torch.Tensor: input sentence queue.
        dp_index: a NumPy array of shape (bsz,1) representing the index of the cluster centre corresponding to each sentence
        """
        # Add new samples to the queue corresponding to the centre of mass
        device = centroid.device
        for j in range(dp_index.size(0)):
            self.enqueue(dp_index[j].item(), centroid[j])

    def uniformity_regularization(self, embeddings):
        # Calculate the distance between samples
        # pairwise_distances = torch.cdist(embeddings, embeddings)
        # Calculate the cosine distance between samples
        pairwise_distances = 1 - self.sim(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
        # Create a diagonal matrix where the values on the diagonal are True and the rest are False
        diag_mask = torch.eye(pairwise_distances.size(0), dtype=torch.bool, device=pairwise_distances.device)
        # Use torch.where to replace the value on the diagonal
        # If the value in diag_mask is True (i.e. on the diagonal), choose float(‘inf’), otherwise choose the original value in pairwise_distances
        pairwise_distances = torch.where(diag_mask, float('inf') * torch.ones_like(pairwise_distances),
                                         pairwise_distances)

        min_distances, _ = torch.min(pairwise_distances, dim=1)
        # Take negative because we want to maximise these distances
        return -torch.mean(min_distances)

    def Gaussian_negative(self, z: torch.Tensor, indexs: torch.Tensor):
        # Assume that z has shape (batch_size, 768) and indexes have shape (batch_size,)
        assert z.shape[0] == indexs.shape[0]
        batch_size, embedding_dim = z.shape
        negative_samples_per_item = 64

        # Generate negative samples for each second centre of mass
        z_negative = torch.zeros((batch_size, negative_samples_per_item, embedding_dim), device=z.device)
        for i in range(batch_size):
            centroid = self.centroid.data[indexs[i]].detach()  # 获取中心点数据
            z_negative[i] = centroid.repeat(negative_samples_per_item, 1) + self.model_args.guss_weight* torch.randn_like(
                centroid.repeat(negative_samples_per_item, 1))

        # Create an optimiser to optimise all negative samples
        z_negative = z_negative.clone().requires_grad_(True)
        optimizer = optim.Adam([z_negative], lr=1e-3)

        for step in range(10):
            optimizer.zero_grad()
            z_expanded = z.unsqueeze(1).expand(-1, negative_samples_per_item, -1)
            sim = self.sim(z_expanded, z_negative)
            uniformity_regs = torch.stack([self.uniformity_regularization(z_negative[i]) for i in range(batch_size)])
            uniformity_reg = uniformity_regs.mean()
            # Take the difference between the mean of the similarity and the uniformity regularisation term
            loss = - (sim.mean() - uniformity_reg)
            loss.backward(retain_graph=True)
            optimizer.step()

        return sim.detach()

        
