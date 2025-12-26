# Localizing Knowledge in Diffusion Transformers

🌐 [Project](https://armanzarei.github.io/Localizing-Knowledge-in-DiTs/) &nbsp;| 📄 [ArXiv](https://arxiv.org/abs/2505.18832)

## 🚩 Updates
- Later in our experiments, we found that using `m_mult_v` in `calc_attn_cont_fn_variant` yields a stronger localization signal—that is, using $\|m \times v\|$ instead of $\|m \times v \times W_o\|$.

## 🔎 Localization and Intervention

Run localization on the 𝓛oc𝓚 dataset and generate samples with intervention:

```bash
python localization_and_intervention/localize_knowledge_and_intervene.py \
    --results_path="{results_path}" \
    --disable_k_dominant_blocks={k} \
    --num_workers=20 \
    --worker_idx={worker_idx} \
    --knowledge_type="style" or "place" or "copyright" or "animal" or "celebrity" or "safety" \
    --model="pixart" or "sana"
```

and for the FLUX model:

```bash
python localization_and_intervention/localize_knowledge_and_intervene_flux.py \
    --results_path="{results_path}" \
    --disable_k_dominant_blocks={k} \
    --num_workers=20 \
    --worker_idx={worker_idx} \
    --knowledge_type={knowledge_type} 
```

These scripts also evaluate the intervention using CLIP scores. For LLaVA-based evaluation, please refer to the sections below.

### Localize a knowledge of your choice

1. Define a dataset containing prompts that represent the target knowledge using the `BaseDataset` class in `dataset.py`. (You can refer to existing examples such as `PlacesDatasetPlacesDataset`, which is designed for localizing place-related knowledge)

2. Use the `load_pipe` function in `loader.py` to load your desired model and pipeline..

3. Use the `localize_dominant_blocks` function from one of the following scripts to perform the localization:

    - `localization_and_intervention/localize_knowledge_and_intervene.py` 
    - `localization_and_intervention/localize_knowledge_and_intervene_flux.py` (for FLUX)

---

Run baseline generations on knowledge-agnostic prompts to assess the impact of the localization process:

```bash
python localization_and_intervention/knowledge_agnostic_gen_and_eval.py \
    --results_path="{results_path}" \
    --model={model_name} \
    --knowledge_type={knowledge_type}
```

Run baseline generations on knowledge prompts without intervention to assess localization:

```bash
python localization_and_intervention/full_knowledge_gen_and_eval.py \
    --results_path="{results_path}" \
    --model={model_name} \
    --knowledge_type={knowledge_type} \
    --worker_idx={worker_idx} \
    --num_workers={num_workers}
```

These scripts also evaluate the intervention using CLIP scores. For LLaVA-based evaluation, please refer to the section below.

## 📊 Evaluation

### 🤖 LLaVA Evaluation

```bash
python llava_eval.py \
    --results_path="{results_path}" \
    --eval_type=eval_all_knowledge_directories \
    --model_name={model_name} \
    --knowledge_type={knowledge_type}
```

set `--eval_type` to `eval_a_single_no_knowledge_directory` to evaluate *"No Knowledge"* generations, where the `results_path` points to a directory containing images (not knowledge subdirectories).


### 🎨 CSD Evaluation for Style

First clone the model checkpoint:

```bash
mkdir pretrained_models
gdown 1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46 -O "pretrained_models/csd_checkpoint.pth"
```

then for the evaluation:

```bash
python csd/csd_calc.py \
    --results_path="your_desired_path_containing_the_images" \
    --artists_list_for_model="pixart" \
    --eval_single_artist_directory
```

Remove `--eval_single_artist_directory` to evaluate each artist's directory in the `results_path`

## 🗂️ Dataset

Please refer to the `dataset/` folder for the 𝓛oc𝓚 dataset, which includes all knowledge categories along with their corresponding prompts for localization. You can also use the data classes defined in `dataset.py` to load and work with the dataset.

## Citation

If you find this useful for your research, please cite the following:
```bibtex
@article{zarei2025localizing,
  title={Localizing Knowledge in Diffusion Transformers},
  author={Zarei, Arman and Basu, Samyadeep and Rezaei, Keivan and Lin, Zihao and Nag, Sayan and Feizi, Soheil},
  journal={arXiv preprint arXiv:2505.18832},
  year={2025}
}
```