from enum import Enum, auto
import math
import torch
from typing import Optional
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class CachingAttnProcessor:
    def __init__(self, idx):
        self.idx = idx
        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []
    
    def clear_maps(self):
        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        with torch.no_grad():
            self.in_norms.append(hidden_states[1].detach().cpu())

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attention_maps.append(attention_probs.detach().cpu())
        self.values.append(value.detach().cpu())

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        with torch.no_grad():
            self.out_norms.append(torch.norm(hidden_states[1], dim=1).detach().cpu())

        return hidden_states


class EmbeddingModifierAttnProcessor:
    def __init__(self, idx, modify_emb=False, print_log=False):
        self.idx = idx
        self.modify_emb = modify_emb
        self.print_log = print_log
    
    def set_clean_emb_and_mask(self, clean_emb, clean_mask):
        self.clean_emb = clean_emb
        self.clean_mask = clean_mask

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        clean_emb = None,
        clean_mask = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        encoder_hidden_states = encoder_hidden_states.clone()
        attention_mask = attention_mask.clone()

        if self.modify_emb:
            if clean_emb is None and self.clean_emb is not None:
                clean_emb = self.clean_emb
                clean_mask = self.clean_mask
            assert clean_emb is not None and clean_mask is not None
            
            encoder_hidden_states[1] = clean_emb
            attention_mask[1] = clean_mask
            if self.print_log:
                print(f"Replaced for block {self.idx}")


        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

class FluxCachingAttnProcessor:
    def __init__(self, idx, max_text_tokens_len=512):
        self.idx = idx
        self.max_text_tokens_len = max_text_tokens_len

        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []
    
    def clear_maps(self):
        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        assert attention_mask is None, "Attention mask is not supported in this processor."
        # assert encoder_hidden_states is not None, "Encoder hidden states are required in this processor."

        with torch.no_grad():
            self.in_norms.append(hidden_states[0].detach().cpu())

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scale_factor = 1 / math.sqrt(query.size(-1))
        attention_probs = query @ key.transpose(-2, -1) * scale_factor
        attention_probs = torch.softmax(attention_probs, dim=-1)

        self.attention_maps.append(attention_probs[:, :, self.max_text_tokens_len:, :self.max_text_tokens_len].detach().cpu())    
        self.values.append(value[:, :, :self.max_text_tokens_len, :].detach().cpu())

        hidden_states = attention_probs @ value
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            with torch.no_grad():
                self.out_norms.append(torch.norm(hidden_states[0], dim=1).detach().cpu())

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
        

class FluxAttnContCalculatorMode(Enum):
    M = "m"
    M_MULT_V = "m_mult_v"
    M_MULT_V_MULT_O = "m_mult_v_mult_o"


class FluxAttnContCalculatorProcessor:
    def __init__(self, token_indices_for_attn_cont_calc, max_text_tokens_len=512, mode: FluxAttnContCalculatorMode = FluxAttnContCalculatorMode.M_MULT_V):
        self.max_text_tokens_len = max_text_tokens_len

        self.token_indices_for_attn_cont_calc = token_indices_for_attn_cont_calc

        self.mode = mode

        self.attn_contribution = 0.
        self.attn_contribution_update_count = 0
    
    def calc_attn_cont(self, attention_probs, value, query, attn):
        assert isinstance(self.token_indices_for_attn_cont_calc, list)
        assert len(self.token_indices_for_attn_cont_calc) > 0

        if self.mode == FluxAttnContCalculatorMode.M_MULT_V_MULT_O:
            m = attention_probs[:, :, self.max_text_tokens_len:, :self.max_text_tokens_len].detach().clone()
            m = m[0, :, :, self.token_indices_for_attn_cont_calc]
            v = value[:, :, :self.max_text_tokens_len, :].detach().clone()
            v = v[0, :, self.token_indices_for_attn_cont_calc, :]
            o = (m @ v).transpose(0, 1).reshape(m.shape[1], v.shape[0] * v.shape[2]) # (24, 1024, 128) -> (1024, 24*128)
            o = o.to(query.dtype)
            if attn.to_out:
                o = attn.to_out[0](o) # (1024, 24*128) -> (1024, 24*128)
            attn_cont = torch.norm(o.to(torch.float32), dim=1).mean().item()
        elif self.mode == FluxAttnContCalculatorMode.M_MULT_V:
            m = attention_probs[:, :, self.max_text_tokens_len:, :self.max_text_tokens_len].detach().clone()
            m = m[0, :, :, self.token_indices_for_attn_cont_calc]
            v = value[:, :, :self.max_text_tokens_len, :].detach().clone()
            v = v[0, :, self.token_indices_for_attn_cont_calc, :]
            o = (m @ v).transpose(0, 1).reshape(m.shape[1], v.shape[0] * v.shape[2]) # (24, 1024, 128) -> (1024, 24*128)
            attn_cont = torch.norm(o.to(torch.float32), dim=1).mean().item()
        elif self.mode == FluxAttnContCalculatorMode.M:
            m = attention_probs[:, :, self.max_text_tokens_len:, :self.max_text_tokens_len].detach().clone()
            m = m[0, :, :, self.token_indices_for_attn_cont_calc] # 24, 1024, 4
            ## torch.all(attn.batch_to_head_dim(m)[0] == m.transpose(0, 1).reshape(m.shape[1], m.shape[0]*m.shape[2])) -> True ##
            attn_cont = torch.mean(torch.norm(attn.batch_to_head_dim(m)[0], dim=1)).item()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return attn_cont

    def update_attn_cont(self, attn_cont):
        self.attn_contribution += attn_cont
        self.attn_contribution_update_count += 1

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        assert attention_mask is None, "Attention mask is not supported in this processor."
        # assert encoder_hidden_states is not None, "Encoder hidden states are required in this processor."

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scale_factor = 1 / math.sqrt(query.size(-1))
        attention_probs = query @ key.transpose(-2, -1) * scale_factor
        attention_probs = torch.softmax(attention_probs, dim=-1)

        self.update_attn_cont(self.calc_attn_cont(attention_probs, value, query, attn))

        hidden_states = attention_probs @ value
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxEmbeddingModifierAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""
    class ProcessorMode(Enum):
        NONE = auto()
        SAVE_ENCODER_HIDDEN_STATES = auto()
        REPLACE_ENCODER_HIDDEN_STATES = auto()

    def __init__(self, mode: ProcessorMode = ProcessorMode.NONE):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.mode = mode
        self.saved_encoder_hidden_states = None
    
    def clear_cache(self):
        self.saved_encoder_hidden_states = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None and self.mode == self.ProcessorMode.SAVE_ENCODER_HIDDEN_STATES:
            self.saved_encoder_hidden_states = hidden_states[:, :512, :].clone()
        elif encoder_hidden_states is None and self.mode == self.ProcessorMode.REPLACE_ENCODER_HIDDEN_STATES:
            hidden_states[:, :512, :] = self.saved_encoder_hidden_states
            
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            if self.mode == self.ProcessorMode.SAVE_ENCODER_HIDDEN_STATES:
                self.saved_encoder_hidden_states = encoder_hidden_states.clone()
            elif self.mode == self.ProcessorMode.REPLACE_ENCODER_HIDDEN_STATES:
                encoder_hidden_states = self.saved_encoder_hidden_states

            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
