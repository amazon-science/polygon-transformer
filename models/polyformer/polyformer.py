# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PolyFormer
"""
from typing import Optional

import logging

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .unify_transformer import TransformerModel

logger = logging.getLogger(__name__)


@register_model("polyformer")
class PolyFormerModel(TransformerModel):
    __jit_unused_properties__ = ["supported_targets"]

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

    @staticmethod
    def add_args(parser):
        super(PolyFormerModel, PolyFormerModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-classifier",
            type=str,
            choices=['mlp', 'linear'],
            help="type of pooler classifier",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        src_tokens,
        src_lengths,
        att_masks,
        prev_output_tokens_11,
        prev_output_tokens_12,
        prev_output_tokens_21,
        prev_output_tokens_22,
        delta_x1,
        delta_y1,
        delta_x2,
        delta_y2,
        patch_images: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        code_masks: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            att_masks=att_masks,
            patch_images=patch_images,
            patch_masks=patch_masks,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
            sample_patch_num=sample_patch_num
        )
        x_cls, x_reg, extra = self.decoder(
            prev_output_tokens_11,
            prev_output_tokens_12,
            prev_output_tokens_21,
            prev_output_tokens_22,
            delta_x1,
            delta_y1,
            delta_x2,
            delta_y2,
            code_masks=code_masks,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return x_cls, x_reg, extra

    def upgrade_state_dict_named(self, state_dict, name):
        pass


@register_model_architecture("polyformer", "polyformer_l")
def polyformer_l_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.0)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_classifier = getattr(args, "pooler_classifier", "mlp")

    args.resnet_drop_path_rate = getattr(args, "resnet_drop_path_rate", 0.0)
    args.encoder_drop_path_rate = getattr(args, "encoder_drop_path_rate", 0.0)
    args.decoder_drop_path_rate = getattr(args, "decoder_drop_path_rate", 0.0)

    args.vis_encoder_type = getattr(args, "vis_encoder_type", "swin-large")
    args.out_index = getattr(args, "out_index", 3)
    args.token_bucket_size = getattr(args, "token_bucket_size", 256)
    args.image_bucket_size = getattr(args, "image_bucket_size", 42)

    args.freeze_encoder_embedding = getattr(args, "freeze_encoder_embedding", False)
    args.freeze_decoder_embedding = getattr(args, "freeze_decoder_embedding", False)
    args.add_type_embedding = getattr(args, "add_type_embedding", True)
    args.attn_scale_factor = getattr(args, "attn_scale_factor", 2)

    args.code_image_size = getattr(args, "code_image_size", 128)
    args.patch_layernorm_embedding = getattr(args, "patch_layernorm_embedding", True)
    args.code_layernorm_embedding = getattr(args, "code_layernorm_embedding", True)
    args.entangle_position_embedding = getattr(args, "entangle_position_embedding", False)
    args.disable_entangle = getattr(args, "disable_entangle", False)
    args.sync_bn = getattr(args, "sync_bn", False)

    args.scale_attn = getattr(args, "scale_attn", False)
    args.scale_fc = getattr(args, "scale_fc", False)
    args.scale_heads = getattr(args, "scale_heads", False)
    args.scale_resids = getattr(args, "scale_resids", False)


@register_model_architecture("polyformer", "polyformer_b")
def polyformer_b_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.out_index = getattr(args, "out_index", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.vis_encoder_type = getattr(args, "vis_encoder_type", "swin-base")
    polyformer_l_architecture(args)