import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
import inspect

import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import find_substring_token_indices, latents_to_images, get_worker_list_chunk, print_arguments
from attention_processor import CachingAttnProcessor, EmbeddingModifierAttnProcessor
from calc_attn_cont_varients import calc_attn_cont_function_variants_mapping, get_calc_attn_cont_fn
from clip_score import get_clip_score
from dataset import get_knowledge_dataset_class_and_get_list_fn, get_eval_text_for_knowledge
from pipelines.loader import load_pipe


def flush_attn_maps_cache(pipe):
    for block in pipe.transformer.transformer_blocks:
        block.attn2.processor.clear_maps()


@torch.no_grad()
def encode_prompt(args, pipe, prompt):
    if args.model == "pixart":
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)
    elif args.model == "sana":
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
            prompt, complex_human_instruction=inspect.signature(pipe.__call__).parameters["complex_human_instruction"].default
        )
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    return prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask


def localize_dominant_blocks(args, pipe, dataset, calc_attn_cont_fn):
    for idx, basic_transformer_block in enumerate(pipe.transformer.transformer_blocks):
        basic_transformer_block.attn2.set_processor(CachingAttnProcessor(idx))

    aggergated_attn_cont = torch.zeros(len(pipe.transformer.transformer_blocks), args.num_timesteps)

    for prompt in tqdm(dataset, desc="Localizing Dominant Blocks"):
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = encode_prompt(args, pipe, prompt)
        
        flush_attn_maps_cache(pipe)

        with torch.no_grad():
            _ = pipe(
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                num_images_per_prompt=1,
                output_type="latent",
                generator=torch.Generator().manual_seed(args.generator_seed),
            )

        aggergated_attn_cont += calc_attn_cont_fn(args, pipe, find_substring_token_indices(prompt, dataset.knowledge, pipe.tokenizer, args.model)) # TODO: Check for overflow 

        flush_attn_maps_cache(pipe)
    
    aggergated_attn_cont = aggergated_attn_cont / len(dataset)
    aggergated_attn_cont = aggergated_attn_cont.mean(dim=1)

    torch.save(aggergated_attn_cont, f"{args.results_path}/{dataset.knowledge}/attn_cont.pt")

    top_dominant_blocks = aggergated_attn_cont.topk(args.disable_k_dominant_blocks).indices

    return top_dominant_blocks


def evaluate(args, pipe, clip_score_fn, dataset, top_dominant_blocks):
    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        block.attn2.set_processor(EmbeddingModifierAttnProcessor(idx, idx in top_dominant_blocks))
    
    res = {}
    for prompt in tqdm(dataset, desc="Evaluating"):
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = encode_prompt(args, pipe, prompt)

            clean_emb, clean_mask, _, _ = encode_prompt(args, pipe, dataset.get_clean_prompt(prompt))
            clean_emb = pipe.transformer.caption_projection(clean_emb)
            if args.model == "pixart":
                clean_emb = clean_emb
            elif args.model == "sana":
                clean_emb = pipe.transformer.caption_norm(clean_emb)
            else:
                raise ValueError(f"Model {args.model} not recognized.")
            clean_mask = (1 - clean_mask.to(torch.float16)) * -10000.0 # TODO: Check for dtype
            clean_mask = clean_mask.unsqueeze(1)

        for i in range(args.num_images_per_eval_prompt):
            seed = args.generator_seed + i

            with torch.no_grad():
                latents = pipe(
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    num_images_per_prompt=1,
                    output_type="latent",
                    cross_attention_kwargs={"clean_emb": clean_emb, "clean_mask": clean_mask},
                    generator=torch.Generator().manual_seed(seed),
                ).images

            image = latents_to_images(pipe, latents)[0].resize((512, 512))

            file_name = f"{prompt}_{seed}.png"

            image.save(f"{args.results_path}/{dataset.knowledge}/{file_name}")
            
            res[file_name] = clip_score_fn(get_eval_text_for_knowledge(args.knowledge_type, dataset.knowledge), image)
    
    with open(f"{args.results_path}/{dataset.knowledge}/results.json", "w") as f:
        json.dump(res, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--generator_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--disable_k_dominant_blocks",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num_images_per_eval_prompt",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--worker_idx",
        type=int,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--calc_attn_cont_fn_variant",
        type=str,
        default="m_mult_v", # m_mult_v_mult_o
        choices=calc_attn_cont_function_variants_mapping.keys(),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default=None,
        choices=["pixart", "sana"],
    )
    parser.add_argument(
        "--knowledge_type",
        type=str,
        required=True,
        default=None,
        choices=["style", "place", "copyright", "animal", "celebrity", "safety"]
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    
    print_arguments(args)

    dataset_class, get_knowledge_list_fn = get_knowledge_dataset_class_and_get_list_fn(args.knowledge_type, for_model=args.model)

    worker_knowledge_list = get_worker_list_chunk(get_knowledge_list_fn(), args.num_workers, args.worker_idx)

    if not os.path.exists(args.results_path):
        print(f"Creating path for {args.results_path}")
        os.makedirs(args.results_path, exist_ok=True)

    pipe = load_pipe(model_name=args.model)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

    calc_attn_cont_fn = get_calc_attn_cont_fn(args.calc_attn_cont_fn_variant)
        
    for knowledge in worker_knowledge_list:
        print(f"Start Localizing for {args.knowledge_type.title()}: {knowledge}")

        if os.path.exists(f"{args.results_path}/{knowledge}/results.json"):
            print(f"Results already exists for {knowledge}. Skipping...")
            continue
        
        if not os.path.exists(f"{args.results_path}/{knowledge}"):
            os.makedirs(f"{args.results_path}/{knowledge}")

        top_dominant_blocks = localize_dominant_blocks(args, pipe, dataset_class(knowledge, "train"), calc_attn_cont_fn)

        print(f"Most dominant blocks for {args.knowledge_type} \"{knowledge}\": {top_dominant_blocks}")

        print(f"Start Evaluation for {args.knowledge_type.title()}: {knowledge}")
        evaluate(args, pipe, partial(get_clip_score, clip_model, clip_processor), dataset_class(knowledge, "both"), top_dominant_blocks)
