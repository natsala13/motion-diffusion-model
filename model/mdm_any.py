from typing import Optional

import clip
import einops
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model.mdm import MDM, OutputProcess, InputProcess, PositionalEncoding, TimestepEmbedder
from model.rotation2xyz import Rotation2xyz
from model.transformer_attention import AttentionStoreEmbeddingsTransformerEncoder, AttentionInjectLayer, AttentionInjectEmbeddingsTransformerEncoder
from data_loaders.tensors import collate


CLIP_VERSION = 'ViT-B/32'

class MdmBase(nn.Module):
    def __init__(self, latent_dim=1024, clip_dim=512, cond_mask_prob=0.1, data_rep='rot6d',
                  dropout=0.1, input_features=263, window_size=-1):
        super().__init__()

        self.dataset = 'interhuman'

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)

        
        self.cond_mask_prob = cond_mask_prob
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.input_process = InputProcess(data_rep, input_features, latent_dim)

        self._temporal_mask = self._create_temporal_mask(window_size=window_size, max_sequence=301)  # TODO: Change to param
        self.embed_text = nn.Linear(clip_dim, latent_dim)
        print('EMBED TEXT')
        print('Loading CLIP...')
        self.clip_model = self.load_and_freeze_clip(CLIP_VERSION)

        self.output_process = OutputProcess(data_rep, input_features, latent_dim, input_features, 1)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset='interhuman')

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    @staticmethod
    def load_and_freeze_clip(clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    @staticmethod
    def _create_temporal_mask(window_size: int, max_sequence: int, attend_to_condition: bool=True) -> Optional[Tensor]:
        '''create a mask of size (seq_len, seq_len) where only the window_size diagonal is one'''
        if window_size < 0:
            return None
        
        window = window_size ** 2
        temporal_masks = torch.stack([torch.Tensor([(i - j) ** 2 < window 
                    for i in range(max_sequence)]) for j in range(max_sequence)])
        
        if attend_to_condition:
            # The name is misleading since we are also attenting to the first frame.
            temporal_masks[:, :2] = 1
        
        return temporal_masks.to('cuda')
    
    def temporal_mask(self, x: torch.Tensor) -> Optional[Tensor]:
        if self._temporal_mask is not None:
            return self._temporal_mask[:x.shape[0], :x.shape[0]]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        # TODO: What happens here? does this feature apply??
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def _apply(self, fn):
        self.rot2xyz.smpl_model._apply(fn)
        return super()._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)

    def _embedd_time_and_text(self, timesteps: Tensor, y: dict) -> Tensor:
        emb: Tensor = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        enc_text = self.encode_text(y['text'])
        emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        return emb
    
    def _create_sequence_from_motion(self, x: Tensor, timesteps, y: dict) -> Tensor:
        embeddings = self._embedd_time_and_text(timesteps, y)
        x = self.input_process(x)
        
        # adding the timestep embed
        xseq = torch.cat((embeddings, x), dim=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)

        return xseq

    def _create_mask(self, batch: int, device: torch.device, y: dict) -> Tensor:
        
        mask = torch.logical_not(y['mask'].squeeze(1).squeeze(1)) # b x seqlen

        step_mask = torch.zeros((batch, 1), dtype=torch.bool, device=device)
        mask = torch.cat([step_mask, mask], dim=1) # b x seqlen + 1
        
        return mask

    @staticmethod
    def save_attention_matrices(attention, timestemp: int):
        if timestemp % 10 != 0:
            return
        
        print(f'Saving Attention figure at timestep {timestemp}')
        for i, att in enumerate(attention):
            # np.save(f'attention/attention_t{timestemp}_l{i}.npy', att[0].detach().cpu().numpy())
            plt.imshow(att[0].detach().cpu().numpy())
            plt.savefig(f'attention/attention_t{timestemp}_l{i}.png')


class MdmActor(MdmBase):
    '''MDM model that returns the motion's embeddings at every step of the transformer.'''
    def __init__(self, latent_dim=1024, clip_dim=512, cond_mask_prob=0.1, ff_size=2048, dropout=0.1,
                input_features=263, num_heads=8, num_layers=8, activation="gelu", window_size=-1):
        super().__init__(latent_dim, clip_dim, cond_mask_prob, input_features=input_features, window_size=window_size)

        attention_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                     nhead=num_heads,
                                                     dim_feedforward=ff_size,
                                                     dropout=dropout,
                                                     activation=activation)

        self.model = AttentionStoreEmbeddingsTransformerEncoder(attention_layer, num_layers=num_layers)

    def forward(self, x: Tensor, timesteps, y: dict) -> tuple[Tensor, list[Tensor]]:
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        xseq = self._create_sequence_from_motion(x, timesteps, y)
        mask = self._create_mask(x.shape[0], x.device, y)
        temporal_mask = self.temporal_mask(xseq)

        output, embeddings = self.model(xseq, mask=temporal_mask, src_key_padding_mask=mask)[1:]  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output, embeddings


class MdmAny(MdmBase):
    '''MDM model that runs actor MDM for single motion, then inject every embedding layer into the self atttention of a second MDM to generate reactor motion.'''
    REFERENCE_CHECKPOINT = 'save/solo5/model000200000.pt'
    FEATURES = 262

    def __init__(self, latent_dim=1024, clip_dim=512, cond_mask_prob=0.1, ff_size=2048, dropout=0.1,
                  input_features=263, num_heads=8, num_layers=8, activation="gelu", window_size=-1):
        
        super().__init__(latent_dim, clip_dim, cond_mask_prob, input_features=input_features, window_size=window_size)

        # self.actor_model = self._mdm_to_store_encoder(actor_model)
        self.actor_model, checkpoint_keys = self._load_reference_model()
        attention_layer = AttentionInjectLayer(d_model=latent_dim,
                                                     nhead=num_heads,
                                                     dim_feedforward=ff_size,
                                                     dropout=dropout,
                                                     activation=activation)

        self.model = AttentionInjectEmbeddingsTransformerEncoder(attention_layer, num_layers=num_layers)
        self.model.load_state_dict(checkpoint_keys)

    def _mdm_to_store_encoder(self, mdm: MDM):
        encoder = mdm.seqTransEncoder
        model = AttentionStoreEmbeddingsTransformerEncoder(encoder.layers[0], num_layers=encoder.num_layers)

        model.load_state_dict(encoder.state_dict())

        return model
    
    def _load_reference_model(self):
        layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4, dim_feedforward=2048, dropout=0.1, activation='gelu')
        model = AttentionStoreEmbeddingsTransformerEncoder(layer, num_layers=8)

        checkpoint = torch.load(self.REFERENCE_CHECKPOINT)
        model_keys = {key.replace('seqTransEncoder.', ''): value for key, value in checkpoint.items() if key.startswith('seqTransEncoder')}

        model.load_state_dict(model_keys)

        return model, model_keys
        

    def forward(self, x: Tensor, timesteps, y: dict):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        xseq1 = self._create_sequence_from_motion(x[:, :self.FEATURES], timesteps, y)
        xseq2 = self._create_sequence_from_motion(x[:, self.FEATURES:], timesteps, y)

        mask = self._create_mask(x.shape[0], x.device, y)
        mask2 = einops.repeat(mask, 'b f -> b (two f)', two=2)
        temporal_mask = self.temporal_mask(xseq1)

        with torch.no_grad():
            output1, embeddings = self.actor_model(xseq1, mask=temporal_mask, src_key_padding_mask=mask)

        output2 = self.model(xseq2, embeddings=embeddings, mask=temporal_mask, src_key_padding_mask=mask2)[1:]  # [seqlen, bs, d]

        output = list(map(self.output_process, [output1[1:], output2]))
        return torch.cat(output, dim=1)
