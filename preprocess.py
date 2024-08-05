import gc

import safetensors
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from safetensors.torch import load_file, save_file
import torch
import json
import os

path = "flux1-dev.sft"
print("----Dev----")

if not os.path.exists(path):
    raise FileNotFoundError("Please download flux-dev.sft file to this directory.")

def read_safetensors_metadata(path):
    with open(path, 'rb') as f:
        header_size = int.from_bytes(f.read(8), 'little')
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        metadata = header.get('__metadata__', {})
        return metadata

metadata = read_safetensors_metadata(path)
print(json.dumps(metadata, indent=4)) #show metadata

sd_pruned = dict() #initialize empty dict

print("Redefining dtype as fp8")
state_dict = load_file(path) #load safetensors file
for key in state_dict: #for each key in the safetensors file
    sd_pruned[key] = state_dict[key].to(torch.float8_e4m3fn) #convert to fp8

# save the pruned safetensors file
#save_file(sd_pruned, "flux1-dev-fp8.safetensors", metadata={"format": "pt", **metadata})
# not in diffusers format :(
dtype = torch.float8_e4m3fn

def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight

def convert_flux_transformer_checkpoint_to_diffusers(
    original_state_dict, num_layers, num_single_layers, inner_dim, mlp_ratio=4.0
):
    converted_state_dict = {}

    ## time_text_embed.timestep_embedder <-  time_in
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
        "time_in.in_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
        "time_in.in_layer.bias"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
        "time_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
        "time_in.out_layer.bias"
    )

    ## time_text_embed.text_embedder <- vector_in
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop(
        "vector_in.in_layer.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop(
        "vector_in.in_layer.bias"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop(
        "vector_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop(
        "vector_in.out_layer.bias"
    )

    # guidance
    has_guidance = any("guidance" in k for k in original_state_dict)
    if has_guidance:
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = original_state_dict.pop(
            "guidance_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = original_state_dict.pop(
            "guidance_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = original_state_dict.pop(
            "guidance_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = original_state_dict.pop(
            "guidance_in.out_layer.bias"
        )

    # context_embedder
    converted_state_dict["context_embedder.weight"] = original_state_dict.pop("txt_in.weight")
    converted_state_dict["context_embedder.bias"] = original_state_dict.pop("txt_in.bias")

    # x_embedder
    converted_state_dict["x_embedder.weight"] = original_state_dict.pop("img_in.weight")
    converted_state_dict["x_embedder.bias"] = original_state_dict.pop("img_in.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms.
        ## norm1
        converted_state_dict[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mod.lin.bias"
        )
        ## norm1_context
        converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mod.lin.bias"
        )
        # Q, K, V
        sample_q, sample_k, sample_v = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0
        )
        context_q, context_k, context_v = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
        )
        context_q_bias, context_k_bias, context_v_bias = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )
        # ff img_mlp
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.2.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.2.bias"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.proj.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.proj.bias"
        )

    # single transfomer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear  <- single_blocks.0.modulation.lin
        converted_state_dict[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.modulation.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.modulation.lin.bias"
        )
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        q, k, v, mlp = torch.split(original_state_dict.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
        q_bias, k_bias, v_bias, mlp_bias = torch.split(
            original_state_dict.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.linear2.weight"
        )
        converted_state_dict[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.linear2.bias"
        )

    converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.weight")
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.bias")
    )

    return converted_state_dict

has_guidance = any("guidance" in k for k in sd_pruned)

#args for transformer
num_layers = 19
num_single_layers = 38
inner_dim = 3072
mlp_ratio = 4.0
print("Converting to diffusers")
converted_state_dict = convert_flux_transformer_checkpoint_to_diffusers(sd_pruned, num_layers, num_single_layers, inner_dim, mlp_ratio=mlp_ratio)
print("Loading as diffusers")
transformer = FluxTransformer2DModel(guidance_embeds=has_guidance)
transformer.load_state_dict(converted_state_dict, strict=True)
print("Saving transformer")
transformer.to(dtype).save_pretrained("flux-dev-fp8/transformer")

del transformer
del converted_state_dict
del sd_pruned
del state_dict

gc.collect()
torch.cuda.empty_cache() # probably not needed but you can never be too sure

path = "flux1-schnell.sft"

print("----Schnell----")

if not os.path.exists(path):
    raise FileNotFoundError("Please download flux-schnell.sft file to this directory.")

metadata = read_safetensors_metadata(path)
print(json.dumps(metadata, indent=4)) #show metadata

sd_pruned = dict() #initialize empty dict

print("Redefining dtype as fp8")
state_dict = load_file(path) #load safetensors file
for key in state_dict: #for each key in the safetensors file
    sd_pruned[key] = state_dict[key].to(torch.float8_e4m3fn) #convert to fp8

has_guidance = any("guidance" in k for k in sd_pruned)

#args for transformer
num_layers = 19
num_single_layers = 38
inner_dim = 3072
mlp_ratio = 4.0
print("Converting to diffusers")
converted_state_dict = convert_flux_transformer_checkpoint_to_diffusers(sd_pruned, num_layers, num_single_layers, inner_dim, mlp_ratio=mlp_ratio)
print("Loading as diffusers")
transformer = FluxTransformer2DModel(guidance_embeds=has_guidance)
transformer.load_state_dict(converted_state_dict, strict=True)
print("Saving transformer")
transformer.to(dtype).save_pretrained("flux-schnell-fp8/transformer")

del transformer
del converted_state_dict
del sd_pruned
del state_dict

gc.collect()
torch.cuda.empty_cache() # probably not needed but you can never be too sure

print("----Autoencoder----")

config = AutoencoderKL.load_config("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae")
vae = AutoencoderKL.from_config(config, scaling_factor=0.3611, shift_factor=0.1159).to(torch.float8_e4m3fn)
print("Loading checkpoint")
try:
    vae_checkpoint = safetensors.torch.load_file("ae.sft")
except:
    print("Failed to load ae.sft! Did you download it?")
print("Converting checkpoint")
converted_vae_state_dict = convert_ldm_vae_checkpoint(vae_checkpoint, vae.config)
print("Loading to diffusers")
vae.load_state_dict(converted_vae_state_dict, strict=True)
print("Saving")
vae.to(dtype).save_pretrained(f"flux-dev-fp8/vae")
vae.to(dtype).save_pretrained(f"flux-schnell-fp8/vae")
