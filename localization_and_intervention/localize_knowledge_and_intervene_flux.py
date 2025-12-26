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

from utils import find_substring_token_indices, get_worker_list_chunk, print_arguments
from attention_processor import FluxAttnContCalculatorProcessor, FluxEmbeddingModifierAttnProcessor, FluxAttnContCalculatorMode
from clip_score import get_clip_score
from dataset import get_knowledge_dataset_class_and_get_list_fn, get_eval_text_for_knowledge
from pipelines.loader import load_pipe


def localize_dominant_blocks(args, pipe, dataset):
    for idx, basic_transformer_block in enumerate(pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks):
        basic_transformer_block.attn.set_processor(FluxAttnContCalculatorProcessor(-1, mode=args.calc_attn_cont_mode))

    aggergated_attn_cont = torch.zeros(len(pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks))

    for prompt in tqdm(dataset, desc="Localizing Dominant Blocks"):
        token_indices = find_substring_token_indices(prompt, dataset.knowledge, pipe.tokenizer, "flux")
        for block in pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks:
            block.attn.processor.token_indices_for_attn_cont_calc = token_indices

        with torch.no_grad():
            _ = pipe(
                prompt=prompt,
                output_type="latent",
                generator=torch.Generator().manual_seed(args.generator_seed),
            )
    
    for idx, block in enumerate(pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks):
        aggergated_attn_cont[idx] = block.attn.processor.attn_contribution / block.attn.processor.attn_contribution_update_count

    torch.save(aggergated_attn_cont, f"{args.results_path}/{dataset.knowledge}/attn_cont.pt")

    top_dominant_blocks = aggergated_attn_cont.topk(args.disable_k_dominant_blocks).indices

    mm_attn_embedding_modifier_indices = []
    single_attn_embedding_modifier_indices = []

    for idx in top_dominant_blocks.tolist():
        if idx < len(pipe.transformer.transformer_blocks):
            mm_attn_embedding_modifier_indices.append(idx)
        else:
            single_attn_embedding_modifier_indices.append(idx - len(pipe.transformer.transformer_blocks))

    return mm_attn_embedding_modifier_indices, single_attn_embedding_modifier_indices


def evaluate(args, pipe, clip_score_fn, dataset, mm_top_dominant_blocks, single_top_dominant_blocks):
    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        block.attn.set_processor(FluxEmbeddingModifierAttnProcessor())
    
    res = {}
    for prompt in tqdm(dataset, desc="Evaluating"):
        for i in range(args.num_images_per_eval_prompt):
            seed = args.generator_seed + i

            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    clean_prompt=dataset.get_clean_prompt(prompt),
                    mm_attn_embedding_modifier_indices=mm_top_dominant_blocks,
                    single_attn_embedding_modifier_indices=single_top_dominant_blocks,
                    replace_pooled_prompt_embeds=True,
                    generator=torch.Generator().manual_seed(seed),
                ).images[0].resize((512, 512))

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
        default=28,
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
        choices=["m_mult_v_mult_o", "m_mult_v", "m"],
    )
    parser.add_argument(
        "--knowledge_type",
        type=str,
        required=True,
        default=None,
        choices=["style", "place", "copyright", "animal", "celebrity", "safety"]
    )
    parser.add_argument(
        "--calc_attn_cont_mode",
        type=str,
        default=FluxAttnContCalculatorMode.M_MULT_V_MULT_O.name.lower(),
        choices=[e.name.lower() for e in FluxAttnContCalculatorMode],
    )

    args = parser.parse_args()

    args.calc_attn_cont_mode = FluxAttnContCalculatorMode[args.calc_attn_cont_mode.upper()]

    return args


if __name__ == '__main__':
    args = parse_args()
    
    print_arguments(args)

    dataset_class, get_knowledge_list_fn = get_knowledge_dataset_class_and_get_list_fn(args.knowledge_type, for_model="flux")

    worker_knowledge_list = get_worker_list_chunk(get_knowledge_list_fn(), args.num_workers, args.worker_idx)

    if not os.path.exists(args.results_path):
        print(f"Creating path for {args.results_path}")
        os.makedirs(args.results_path, exist_ok=True)

    pipe = load_pipe("flux")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

    for knowledge in worker_knowledge_list:
        print(f"Start Localizing for {args.knowledge_type.title()}: {knowledge}")

        if os.path.exists(f"{args.results_path}/{knowledge}/results.json"):
            print(f"Results already exists for {knowledge}. Skipping...")
            continue
        
        if not os.path.exists(f"{args.results_path}/{knowledge}"):
            os.makedirs(f"{args.results_path}/{knowledge}")

        mm_top_dominant_blocks, single_top_dominant_blocks = localize_dominant_blocks(args, pipe, dataset_class(knowledge, "train"))

        print(f"Most MM dominant blocks for {args.knowledge_type} \"{knowledge}\": {mm_top_dominant_blocks}")
        print(f"Most single dominant blocks for {args.knowledge_type} \"{knowledge}\": {single_top_dominant_blocks}")

        print(f"Start Evaluation for {args.knowledge_type.title()}: {knowledge}")
        evaluate(args, pipe, partial(get_clip_score, clip_model, clip_processor), dataset_class(knowledge, "both"), mm_top_dominant_blocks, single_top_dominant_blocks)
