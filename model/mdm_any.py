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
from model.transformer_attention import AttentionStoreEmbeddingsTransformerEncoder, AttentionInjectLayer, AttentionInjectEmbeddingsTransformerEncoder, SymetricInjectLayer, DoubleInjectTransformerEncoder
from data_loaders.tensors import collate


CLIP_VERSION = 'ViT-B/32'

class MdmBase(nn.Module):
    def __init__(self, latent_dim=1024, diffusion=None, clip_dim=512, cond_mask_prob=0.1, data_rep='rot6d',
                  dropout=0.1, input_features=263, window_size=-1):
        super().__init__()

        self.dataset = 'interhuman'
        self.diffusion = diffusion
        self.cond_mask_prob = cond_mask_prob

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)

        self.input_features = input_features
        self.input_process = InputProcess(data_rep, input_features, latent_dim)
        self.output_process = OutputProcess(data_rep, input_features, latent_dim, input_features, 1)

        self.embed_text = nn.Linear(clip_dim, latent_dim)
        self.clip_model = self.load_and_freeze_clip(CLIP_VERSION)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset='interhuman')

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    @staticmethod
    def load_and_freeze_clip(clip_version):
        print('Loading CLIP...')
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

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
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
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
        
        mask = torch.logical_not(y['mask'].squeeze(1).squeeze(1)).to(device)

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


    def _create_args_for_generation(self, batch: dict) -> tuple[int, int, dict]:
        batch_size = len(batch['text'])
        n_frames = batch['motion_lens'][0]

        collate_args = [{'inp': torch.zeros(n_frames), 'text': batch['text'][0], 'tokens': None, 'lengths': n_frames}] * batch_size
        _, model_kwargs = collate(collate_args)
        device = next(self.parameters()).device
        model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        return batch_size, n_frames, model_kwargs


    def forward_test(self, batch: dict, **kwargs) -> Tensor:
        assert self.diffusion != None
        batch_size, n_frames, model_kwargs = self._create_args_for_generation(batch)
        
        sample = self.diffusion.p_sample_loop(
            self,
            (batch_size, self.input_features, 1, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        return sample.permute(0, 3, 1, 2)


class MdmAttend(MdmBase):
    '''MDM model that returns the motion's embeddings at every step of the transformer.'''
    class InputAttendProcess(nn.Module):
        NAMES = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
            'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
            'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
            'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe',
            'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe']

        def __init__(self):
            super().__init__()
            self.input_feats = 420
            self.latent_dim = 1024
            self.pose_embedding = nn.Linear(self.input_feats, self.latent_dim)
        
        @staticmethod
        def _rearrane_frame(motion: Tensor) -> Tensor:
            '''motion is a (b, features, 1, frames) rearange it to (b, joints, extended features, frames).'''
            root_data = motion[:, :4, 0, :]
            positions = motion[:, 4: 63 + 4, 0, :]
            rotations = motion[:, 4 + 63: 4 + 63 + 126, 0, :]
            velocity = motion[:, 4 + 63 + 126: 4 + 63 + 126 + 66, 0, :]
            foot_contact = motion[:, -4:, 0, :]

            movement = torch.cat([einops.rearrange(positions, 'b (j f) t -> b j f t', f=3),
                                  einops.rearrange(rotations, 'b (j f) t -> b j f t', f=6)], axis=2) # type: ignore
            movement = F.pad(movement, (0, 0, 0, 0, 0, 1))
            motion = torch.cat([movement, einops.rearrange(velocity, 'b (j f) t -> b j f t', f=3)], axis=2) # type: ignore
            features = motion.shape[-2]
            extendend_motion = torch.cat([F.pad(root_data,(0, 0, 0, 8))[:, None, :, :], motion,
                                                  F.pad(foot_contact[:, :2, :],(0, 0, 0, features - 2))[:, None, :, :],
                                                  F.pad(foot_contact[:, 2:, :],(0, 0, 0, features - 2))[:, None, :, :]], axis=1) # type: ignore
            
            return extendend_motion
        
        @staticmethod
        def _split_to_joint_groups(motion: Tensor) -> Tensor:
            spine_chain = motion[:, :7]
            left_hand = motion[:, 7: 7 + 4]
            right_hand = motion[:, 7 + 4: 7 + 2 * 4]
            left_foot = motion[:, 7 + 2 * 4: 7 + 3 * 4]
            right_foot = motion[:, 7 + 3 * 4: 7 + 4 * 4]

            left_foot = torch.cat((left_foot, motion[:, -2:-1]), dim=1)
            right_foot = torch.cat((right_foot, motion[:, -1:]), dim=1)

            max_features = spine_chain.shape[1] * spine_chain.shape[2]
            all_tokens = [spine_chain, left_hand, right_hand, left_foot, right_foot]
            all_tokens_flatten = [einops.rearrange(token, 'b (one j) f t -> b one (j f) t', one=1) for token in all_tokens]
            all_tokens_padded = [F.pad(token, (0, 0, 0, max_features - token.shape[2])) for token in all_tokens_flatten]

            return torch.cat(all_tokens_padded, dim=1)

        def forward(self, x):
            extended_motion = self._rearrane_frame(x)
            motion_per_joint_group = self._split_to_joint_groups(extended_motion)
            time_patches_motion = einops.rearrange(motion_per_joint_group, 'b j f (five t) -> b j f five t', five=5)
            motion = einops.rearrange(time_patches_motion, 'b j f five t -> (j t) b (f five)')

            motion = self.pose_embedding(motion)
            return motion

    class OutputProcess(nn.Module):
        FOOT_TOKENS = 5
        HAND_TOKENS = 4
        SPINE_TOKENS = 7

        def __init__(self):
            super().__init__()
            self.input_feats = 420
            self.latent_dim = 1024

            self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
                
        def _from_joint_group_to_3d_matrix(self, motion: Tensor) -> Tensor:
            all_tokens_matrix = list(einops.rearrange(motion, 'b j (seven f) t -> j seven f t b', seven=7))
            # all_tokens_padded = list(einops.rearrange(motion, 'b j f t -> j b f t'))
            num_joints_per_token = [self.SPINE_TOKENS, self.HAND_TOKENS, self.HAND_TOKENS, self.FOOT_TOKENS, self.FOOT_TOKENS]
            spine_chain, left_hand, right_hand, left_foot, right_foot = [token[:num_joints] 
                                                                         for token, num_joints in zip(all_tokens_matrix, num_joints_per_token)]

            foot_contact = torch.cat([left_foot[-1:], right_foot[-1:]])
            reconstracted_motion = torch.cat([spine_chain, left_hand, right_hand, left_foot[:-1], right_foot[:-1], foot_contact])
            return reconstracted_motion
        
        def _rearange_matrix_to_features(self, extended_motion: Tensor) -> Tensor:
            root_data = extended_motion[0, :4]
            motion = extended_motion[1:-2]
            foot_contact = einops.rearrange(extended_motion[-2:, :2], 'foot joint t b -> (foot joint) t b')

            velocity = einops.rearrange(motion[:, -3:], 'j f t b -> (j f) t b')
            movement = motion[:-1, :-3]
            rotations = einops.rearrange(movement[:, -6:], 'j f t b -> (j f) t b')
            positions = einops.rearrange(movement[:, :3], 'j f t b -> (j f) t b')

            humanml_representation = torch.cat([root_data, positions, rotations, velocity, foot_contact])

            return einops.rearrange(humanml_representation, 'f t b -> b f t')


        def forward(self, output):
            output = self.poseFinal(output)
            time_patches_motion = einops.rearrange(output, '(j t) b (f five) -> b j f five t', five=5, j=5)
            motion_per_joint_group = einops.rearrange(time_patches_motion, 'b j f five t -> b j f (five t)')
            extended_motion = self._from_joint_group_to_3d_matrix(motion_per_joint_group)
            output = self._rearange_matrix_to_features(extended_motion)

            return output[:, :, None]

    def __init__(self, latent_dim=1024, diffusion=None, clip_dim=512, cond_mask_prob=0.1, ff_size=2048, dropout=0.1,
                input_features=263, num_heads=8, num_layers=8, activation="gelu", window_size=-1):
        super().__init__(latent_dim, diffusion=diffusion, clip_dim=clip_dim,
                          cond_mask_prob=cond_mask_prob, input_features=input_features, window_size=window_size)

        self.input_process = self.InputAttendProcess()
        self.output_process = self.OutputProcess()

        attention_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                     nhead=num_heads,
                                                     dim_feedforward=ff_size,
                                                     dropout=dropout,
                                                     activation=activation)

        self.model = nn.TransformerEncoder(attention_layer, num_layers=num_layers)

    def forward(self, x: Tensor, timesteps, y: dict) -> tuple[Tensor, list[Tensor]]:
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        xseq = self._create_sequence_from_motion(x, timesteps, y)
        mask = self._create_mask(x.shape[0], x.device, y)
        # temporal_mask = self.temporal_mask(xseq)

        output = self.model(xseq, src_key_padding_mask=mask)[1:]

        output = self.output_process(output)

        return output


class MdmAny(MdmBase):
    '''MDM model that runs actor MDM for single motion,
     then inject every embedding layer into the self atttention of a second MDM to generate reactor motion.'''
    REFERENCE_CHECKPOINT = 'save/solo5/model000200000.pt'
    FEATURES = 262

    def __init__(self, latent_dim=1024, diffusion=None, clip_dim=512, cond_mask_prob=0.1, ff_size=2048, dropout=0.1,
                  input_features=263, num_heads=8, num_layers=8, activation="gelu", window_size=-1):
        
        super().__init__(latent_dim, diffusion=diffusion, clip_dim=clip_dim,
                          cond_mask_prob=cond_mask_prob, input_features=input_features, window_size=window_size)

        self.data_rep = 'interhuman any'
        self.njoints = 524
        self.nfeats = 1
        self.input_features = input_features * 2
        self.actor_model, checkpoint_keys, input_keys, output_keys = self._load_reference_model()
        attention_layer = AttentionInjectLayer(d_model=latent_dim,
                                                     nhead=num_heads,
                                                     dropout=dropout,
                                                     activation=activation)

        self.model = AttentionInjectEmbeddingsTransformerEncoder(attention_layer, num_layers=num_layers)
        self.model.load_state_dict(checkpoint_keys)
        self.input_process.load_state_dict(input_keys)
        self.output_process.load_state_dict(output_keys)

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
        input_keys = {key.replace('input_process.', ''): value for key, value in checkpoint.items() if key.startswith('input_process')}
        output_keys = {key.replace('output_process.', ''): value for key, value in checkpoint.items() if key.startswith('output_process')}

        model.load_state_dict(model_keys)

        return model, model_keys, input_keys, output_keys
        

    def forward(self, x: Tensor, timesteps, y: dict):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        xseq1 = self._create_sequence_from_motion(x[:, :self.FEATURES], timesteps, y)
        xseq2 = self._create_sequence_from_motion(x[:, self.FEATURES:], timesteps, y)

        mask = self._create_mask(x.shape[0], x.device, y)
        mask2 = einops.repeat(mask, 'b f -> b (two f)', two=2)
        # temporal_mask = self.temporal_mask(xseq1)

        with torch.no_grad():
            output1, embeddings = self.actor_model(xseq1, src_key_padding_mask=mask)

        output2 = self.model(xseq2, embeddings=embeddings, src_key_padding_mask=mask2)[1:]  # [seqlen, bs, d]

        output = list(map(self.output_process, [output1[1:], output2]))
        return torch.cat(output, dim=1)

class MdmTime(MdmBase):
    '''Run MDM while concatenating two motions on time axis.'''
    class CouplePositionalEncoding(nn.Module):
        '''Positional encoding for concatenating motiion of 2 person on time (token) axis.'''
        def __init__(self, d_model, dropout=0.1, max_len=1000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.max_len = max_len

            pe = torch.zeros(2 * max_len, d_model)
            position = torch.arange(0, 2 * max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)  # So that the tensor wouldn't be backpropagated.

        def forward(self, x):
            # not used in the final model
            one_motion_shape = x.shape // 2
            encoding = torch.cat(self.pe[:one_motion_shape], self.pe[self.max_len:one_motion_shape + self.max_len])
            x = x + encoding

            return self.dropout(x)
        
    def __init__(self, latent_dim=1024, diffusion=None, clip_dim=512, cond_mask_prob=0.1, ff_size=2048, dropout=0.1,
                input_features=262, num_heads=8, num_layers=8, activation="gelu", window_size=-1):
        super().__init__(latent_dim, diffusion=diffusion, clip_dim=clip_dim,
                          cond_mask_prob=cond_mask_prob, input_features=input_features, window_size=window_size)

        self.positional_encoding = self.CouplePositionalEncoding(latent_dim, dropout)
        attention_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                     nhead=num_heads,
                                                     dim_feedforward=ff_size,
                                                     dropout=dropout,
                                                     activation=activation)

        self.model = nn.TransformerEncoder(attention_layer, num_layers=num_layers)
    
    def _create_mask(self, batch: int, device: torch.device, y: dict) -> Tensor:
        
        mask = torch.logical_not(y['mask'].squeeze(1).squeeze(1)).to(device)
        _, tokens = mask.shape
        mask[:, tokens // 2:] = mask[:, :tokens // 2]

        step_mask = torch.zeros((batch, 1), dtype=torch.bool, device=device)
        mask = torch.cat([step_mask, mask], dim=1) # b x seqlen + 1
        
        return mask
    
    def _create_args_for_generation(self, batch: dict) -> tuple[int, int, dict]:
        batch_size = len(batch['text'])
        n_frames = batch['motion_lens'][0]

        collate_args = [{'inp': torch.zeros(n_frames * 2), 'text': batch['text'][0], 'tokens': None, 'lengths': n_frames}] * batch_size
        _, model_kwargs = collate(collate_args)
        device = next(self.parameters()).device
        model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        return batch_size, n_frames * 2, model_kwargs
    
    def forward(self, x: Tensor, timesteps, y: dict) -> tuple[Tensor, list[Tensor]]:
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        xseq = self._create_sequence_from_motion(x, timesteps, y)
        mask = self._create_mask(x.shape[0], x.device, y)

        output = self.model(xseq, src_key_padding_mask=mask)[1:]
        output = self.output_process(output)

        return output
    
    def forward_test(self, batch: dict, **kwargs) -> Tensor:
        # batch['motion_lens'][0] = 400self.max_length
        sample = super().forward_test(batch, **kwargs)
        
        return einops.rearrange(sample, 'b (two t) f one -> b t two (f one)', two=2)



class MdmSimetric(MdmBase):
    '''Run te same model twice while concatenating K matrices from one inference to the second one.'''
    REFERENCE_CHECKPOINT = 'save/solo5/model000200000.pt'
    FEATURES = 262

    def __init__(self, latent_dim=1024, diffusion=None, second_attention=False,
                  clip_dim=512, cond_mask_prob=0.1, ff_size=2048, dropout=0.1,
                  input_features=263, num_heads=8, num_layers=8, activation="gelu", window_size=-1):

        super().__init__(latent_dim, diffusion=diffusion,
                          clip_dim=clip_dim, cond_mask_prob=cond_mask_prob,
                            input_features=input_features, window_size=window_size)

        self.data_rep = 'interhuman any'
        self.njoints = 524
        self.nfeats = 1
        self.input_features = input_features * 2
        checkpoint_keys, input_keys, output_keys = self._load_reference_model()
        attention_layer = SymetricInjectLayer(d_model=latent_dim,
                                                     nhead=num_heads,
                                                     dropout=dropout,
                                                     activation=activation,
                                                     second_attention=second_attention)

        self.model = DoubleInjectTransformerEncoder(attention_layer, num_layers=num_layers)
        self.model.load_state_dict(checkpoint_keys, strict=False)
        self.input_process.load_state_dict(input_keys)
        self.output_process.load_state_dict(output_keys)

    def _load_reference_model(self) -> tuple[dict, dict, dict]:
        checkpoint = torch.load(self.REFERENCE_CHECKPOINT)
        model_keys = {key.replace('seqTransEncoder.', ''): value
                      for key, value in checkpoint.items() if key.startswith('seqTransEncoder')}
        input_keys = {key.replace('input_process.', ''): value
                      for key, value in checkpoint.items() if key.startswith('input_process')}
        output_keys = {key.replace('output_process.', ''): value
                       for key, value in checkpoint.items() if key.startswith('output_process')}

        return model_keys, input_keys, output_keys


    def forward(self, x: Tensor, timesteps, y: dict):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        xseq1 = self._create_sequence_from_motion(x[:, :self.FEATURES], timesteps, y)
        xseq2 = self._create_sequence_from_motion(x[:, self.FEATURES:], timesteps, y)

        mask = self._create_mask(x.shape[0], x.device, y)
        mask2 = einops.repeat(mask, 'b f -> b (two f)', two=2)

        output1, output2 = self.model(xseq1, xseq2, src_key_padding_mask=mask2)  # [seqlen, bs, d]

        output = list(map(self.output_process, [output1[1:], output2[1:]]))
        return torch.cat(output, dim=1)
