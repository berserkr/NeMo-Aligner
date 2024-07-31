import torch

from megatron.core import parallel_state
from pytorch_lightning.trainer.trainer import Trainer
#from pytriton.model_config import ModelConfig
#from pytriton.model_config.common import DynamicBatcher
from pytriton.model_config import Tensor

import threading
from typing import Dict
from typing import List

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.utils.text_generation_utils import tokenize_batch
from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, set_autocast_gpu_dtype

from nemo_aligner.utils.distributed import run_distributed_inference
from nemo_aligner.utils.server_utils import decode_bytes_ndarray, pad_input
from nemo_aligner.utils.server_utils import (
    calculate_inference_batch_padding_multiple,
    pad_batch_and_strip_sequence
)

from tqdm import tqdm, trange
import numpy as np


# reward bench stuff...
from rewardbench import load_eval_dataset
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

from transformers import AutoTokenizer

import logging
import sys


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

SYSTEM_PROMPT_TEMPLATE = "<extra_id_0>System\n{value}\n"

USER_TURN_TEMPLATE = "<extra_id_1>User\n{value}\n"

ASSISTANT_TURN_TEMPLATE = "<extra_id_1>Assistant\n{value}\n"

LABEL_PREFIX = "<extra_id_2>"

OPEN_ASSISTANT_ATTRIBUTES = ["quality", "toxicity", "humor", "creativity"]

HELPSTEER_ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

ALL_STEERLM_ATTRIBUTES = OPEN_ASSISTANT_ATTRIBUTES + HELPSTEER_ATTRIBUTES

REWARD_VECTOR_340B = [0, 0, 0, 0, 0.3, 0.74, 0.46, 0.47, -0.33]
REWARD_VECTOR_70B = [0, 0, 0, 0, 0.65, 0.8, 0.45,0.55, -0.4]
REWARD_VECTOR = REWARD_VECTOR_340B


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = logging.INFO
logger.setLevel(log_level)


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def jsonline_to_turns(jsonl):
    """
    {   "conversations": [
                {"value": "explain master slave replication nsql", "from": "User", "label": null}, 
                {"value": "In the context of NoSQL databases, master-slave replication refers to a configuration where a single master node writes data, and one or more slave nodes read data from the master and replicate it to provide read scalability. The master node is responsible for accepting write requests and updating its own data, while the slave nodes are responsible for replicating the data from the master and serving read requests.\n\nIn this configuration, the master node is the only node that can make          changes to the data, while the slave nodes can only read the data and replicate it. This ensures that the data is consistent across all nodes and that there is no data loss in case of a failure of a single node.\n\nMaster-slave replication in NoSQL databases can be implemented using various technologies, such as Apache Cassandra, MongoDB, and Apache HBase. In each of these databases, the master-slave replication configuration can be set up using specific configuration options and parameters.\n\nIt's worth noting that master-slave replication is not a failover solution, as the failure of the master node will result in the loss of data until the node is brought back online. Therefore, it's important to have a proper disaster recovery plan in place to ensure that data is protected in case of a failure.",  "from": "Assistant", "label": "helpfulness:4,correctness:4,coherence:4,complexity:3,verbosity:2"}
            ], 
            "system": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", 
            "mask": "User", 
            "type": "VALUE_TO_TEXT"
    }
    """
    user = jsonl["mask"]
    turns = []

    text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)

    for turn in jsonl["conversations"]:
        value = turn["value"]
        if turn["from"] == user:
            text += USER_TURN_TEMPLATE.format(value=value)
        else:
            text += ASSISTANT_TURN_TEMPLATE.format(value=value)
        if "label" in turn and turn["label"] is not None:
            out_text = text + LABEL_PREFIX
            turns.append(out_text)

    return turns


def dataloader_to_template(data):
    """
    Input in batches:
    [
        {'content': 
            [
                'How do I detail a car?', 
                'Who created the Superman cartoon character?'
            ], 
            'role': 
            [
                'user', 
                'user'
            ]
        }, 
        {'content': 
            [
                "Detailing a car involves a thorough cleaning inside and out, as well as polishing and waxing to protect the vehicle's surfaces. Here's a step-by-step guide to detailing a car:\n\n**Exterior Detailing:**\n\n1. **Wash the Car:**\n   - Rinse the car with water to remove loose dirt.\n   - Use a car wash soap and microfiber wash mitt to clean the car from top to bottom.\n   - Clean the wheels and tires with a brush and a wheel cleaner.\n   - Rinse the car thoroughly to remove all soap.\n\n2. **Dry the Car:**\n   - Use a microfiber towel or a chamois to dry the car to prevent water spots.\n\n3. **Clay Bar Treatment:**\n   - Use a clay bar with a lubricant to remove embedded surface contaminants from the paint.\n\n4. **Polishing:**\n   - Apply car polish with a dual-action polisher or by hand to correct paint imperfections and create a smooth surface.\n\n5. **Waxing:**\n   - Apply a coat of wax or paint sealant to protect the paint and give it a glossy finish.\n\n6. **Windows and Mirrors:**\n   - Clean the windows and mirrors with a glass cleaner and a microfiber towel.\n\n7. **Tire and Trim Dressing:**\n   - Apply a tire dressing to the tires for a shiny finish.\n   - Use a trim restorer or protectant on plastic and rubber parts to prevent fading.\n\n**Interior Detailing:**\n\n1. **Remove Trash:**\n   - Clear out any trash and remove personal items from the car.\n\n2. **Vacuum:**\n   - Vacuum the seats, carpets, floor mats, and trunk.\n   - Use a brush attachment for the dashboard and door panels.\n\n3. **Shampoo Carpets and Upholstery:**\n   - Use a carpet cleaner and a brush to clean the carpets and upholstery.\n   - For leather interiors, use a leather cleaner and conditioner.\n\n4. **Clean Hard Surfaces:**\n   - Wipe down all hard surfaces (dashboard, center console, door panels, etc.) with a mild all-purpose cleaner and a microfiber cloth.\n\n5. **Windows and Mirrors:**\n   - Clean the interior side of windows and mirrors.\n\n6. **Air Vents and Crevices:**\n   - Use a detailing brush or compressed air to clean out air vents and hard-to-reach crevices.\n\n7. **Final Touches:**\n   - Apply a protectant to the dashboard and other plastic components.\n   - Replace air fresheners if needed.\n\n**Additional Tips:**\n\n- Work in the shade or a cool, well-ventilated garage to prevent products from drying too quickly and leaving residue.\n- Use separate buckets for washing and rinsing to avoid contaminating the clean water with dirt.\n- Always use gentle, non-abrasive materials and cleaners specifically designed for automotive use to avoid damaging surfaces.\n- Move in a systematic way to ensure you don't miss any spots.\n\nBy following these steps, you'll give your car a thorough clean that not only makes it look great but also helps in maintaining its value. Remember, regular detailing can prevent wear and tear and keep your car looking new for years to come.", 
                "Superman, the iconic comic book superhero, was created by writer Jerry Siegel and artist Joe Shuster. Superman first appeared in Action Comics #1, which was published by Detective Comics, Inc. (later DC Comics) in June 1938. The character's immense popularity established him as one of the most enduring and recognizable figures in the superhero genre."
            ], 
            'role': [
                'assistant', 
                'assistant'
            ]
        }
    ]

    {   "conversations": [
                {"value": "explain master slave replication nsql", "from": "User", "label": null}, 
                {"value": "In the context of NoSQL databases, master-slave replication refers to a configuration where a single master node writes data, and one or more slave nodes read data from the master and replicate it to provide read scalability. The master node is responsible for accepting write requests and updating its own data, while the slave nodes are responsible for replicating the data from the master and serving read requests.\n\nIn this configuration, the master node is the only node that can make          changes to the data, while the slave nodes can only read the data and replicate it. This ensures that the data is consistent across all nodes and that there is no data loss in case of a failure of a single node.\n\nMaster-slave replication in NoSQL databases can be implemented using various technologies, such as Apache Cassandra, MongoDB, and Apache HBase. In each of these databases, the master-slave replication configuration can be set up using specific configuration options and parameters.\n\nIt's worth noting that master-slave replication is not a failover solution, as the failure of the master node will result in the loss of data until the node is brought back online. Therefore, it's important to have a proper disaster recovery plan in place to ensure that data is protected in case of a failure.",  "from": "Assistant", "label": "helpfulness:4,correctness:4,coherence:4,complexity:3,verbosity:2"}
            ], 
            "system": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", 
            "mask": "User", 
            "type": "VALUE_TO_TEXT"
    }
    """
    samples = []

    text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)

    assert len(data) == 2

    user_data = data[0]
    assistant_data = data[1]

    turns = []

    batch_size = len(user_data['content'])

    user_content = user_data['content']
    assistant_content = assistant_data['content']

    for i in range(0,batch_size):
        text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)
        text += USER_TURN_TEMPLATE.format(value=user_content[i])
        text += ASSISTANT_TURN_TEMPLATE.format(value=assistant_content[i])
        text += LABEL_PREFIX
        samples.append(text)

    return samples


def get_reward(
    sentences: List[str], infer_fn, tokenize_func, model_forward_micro_batch_size = [2], strip_sequence_length_to_multiple = None
):

    print_rank_0(f'Running get reward with input={sentences}')

    tokens, sequence_lengths = tokenize_func(sentences)
    sequence_lengths = sequence_lengths.unsqueeze(-1)

    pad_batch_to_multiple = calculate_inference_batch_padding_multiple(
        tokens.shape[0], model_forward_micro_batch_size
    )

    inputs, extra, _ = pad_batch_and_strip_sequence(
        tokens,
        sequence_lengths,
        pad_to_multiple=pad_batch_to_multiple,
        strip_sequence_length_to_multiple=strip_sequence_length_to_multiple,
    )
    rewards = run_distributed_inference(inputs=inputs, infer_fn=infer_fn)

    rewards = rewards[: rewards.shape[0] - extra]

    return rewards


def get_scores(infer_fn, tokenize_func, sample_batch, model_forward_micro_batch_size, strip_sequence_length_to_multiple):
    """
    infer_fn: inference funct
    sample_batch: batch of sentences
    """

    print_rank_0(f'Sample: {sample_batch}')

    samples = dataloader_to_template(sample_batch)

    print_rank_0(f'Batched Samles: {samples}')

    reward_batch = get_reward(
        samples, infer_fn=infer_fn, tokenize_func=tokenize_func, model_forward_micro_batch_size=model_forward_micro_batch_size, strip_sequence_length_to_multiple=strip_sequence_length_to_multiple
    )

    print_rank_0(f'{len(reward_batch)} - reward_batch: {reward_batch}')

    labels = []
    scores = []

    for reward in reward_batch:

        if len(reward) == 2: # likely 2d arr
            reward = reward[0]

        # hack, check if we have 5 or 9 scores
        if len(reward) == 9: 
            valid_5 = False
        elif len(reward) == 5:
            valid_5 = True
        else:
            continue

        print_rank_0(f'Processing reward: {reward}')

        reward_each = [min(4.0, max(0.0, float(r))) for r in reward]
        reward_each = [round(r) for r in reward_each]

        if valid_5:
            reward_string = ",".join(f"{a}:{r}" for a, r in zip(HELPSTEER_ATTRIBUTES, reward_each))
            score = sum([REWARD_VECTOR[4:][i] * reward[i] for i in range(0,len(REWARD_VECTOR[4:]))])
        else:
            reward_string = ",".join(f"{a}:{r}" for a, r in zip(ALL_STEERLM_ATTRIBUTES, reward_each))
            score = sum([REWARD_VECTOR[i] * reward[i] for i in range(0,len(REWARD_VECTOR))])
        
        labels.append({'reward_string' : reward_string})
        scores.append({'score' : score})

        #turn["score"] = sum([REWARD_VECTOR_70B[i] * reward[i] for i in range(0,len(REWARD_VECTOR_70B))])
        print_rank_0(f'xxx>>> {reward_string}')
        print_rank_0(f'xxx>>> {score}')

    return labels, scores


@hydra_runner(config_path="conf", config_name="inference_rm")
def main(cfg) -> None:

    cfg.model = load_and_override_model_config(cfg.rm_model_file, cfg.model)
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    set_autocast_gpu_dtype(cfg.trainer.precision)
    if trainer.precision == "16":
        cfg.model.megatron_amp_O2 = False
    elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get("megatron_amp_O2", False):
        cfg.model.megatron_amp_O2 = True

    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    ptl_model = load_from_nemo(reward_model_cls, cfg.model, trainer, strict=True, restore_path=cfg.rm_model_file,)
    ptl_model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        ptl_model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    ptl_model = ptl_model.cuda()

    dp_size = parallel_state.get_data_parallel_world_size()
    #max_batch_size = cfg.inference.micro_batch_size * dp_size
    dp_size = cfg.inference.micro_batch_size * dp_size

    infer_fn = ptl_model.infer
    ptl_model.prepare_for_inference()

    def tokenize_func(sentences):
        return tokenize_batch(
            sentences=sentences,
            tokenizer=ptl_model.tokenizer,
            max_len=ptl_model.cfg.encoder_seq_length,
            add_BOS=False,
            add_EOS=False,
        )

    # now, load datasets...
    print_rank_0("Running core eval dataset.")

    # primary set compiles slightly more information
    dataset, subsets = load_eval_dataset(
        core_set=True,
        conv=None,
        custom_dialogue_formatting=True,
        tokenizer=ptl_model.tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "prompt"],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dp_size,
        shuffle=False,
        drop_last=False,
    )

    print_rank_0(f"BBB - Batch Size: {dp_size}")

    model_forward_micro_batch_size = cfg.model.get("forward_micro_batch_size", cfg.model.micro_batch_size)
    strip_sequence_length_to_multiple = cfg.inference.get("strip_sequence_length_to_multiple", None)

    results = []
    scores_chosen = []
    scores_rejected = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        print_rank_0(f"RM inference step {step}/{len(dataloader)}")
    
        _, rewards_chosen = get_scores(infer_fn, tokenize_func, batch["text_chosen"], model_forward_micro_batch_size=model_forward_micro_batch_size, strip_sequence_length_to_multiple=strip_sequence_length_to_multiple)
        print_rank_0(f'rewards_chosen: {rewards_chosen}')

        _, rewards_rejected = get_scores(infer_fn, tokenize_func, batch["text_rejected"], model_forward_micro_batch_size=model_forward_micro_batch_size, strip_sequence_length_to_multiple=strip_sequence_length_to_multiple)
        print_rank_0(f'rewards_rejected: {rewards_rejected}')

        if len(rewards_chosen)==0 or len(rewards_rejected)==0:
            continue

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            score_chosen_batch = [result["score"] for result in rewards_chosen]
            score_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            score_chosen_batch = rewards_chosen.cpu().numpy().tolist()
            score_rejected_batch = rewards_rejected.cpu().numpy().tolist()

        # log results
        [
            results.append(1) if chosen > rejected else results.append(0)
            for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
        ]
        scores_chosen.extend(score_chosen_batch)
        scores_rejected.extend(score_rejected_batch)

    if len(results) == 0:
         print_rank_0("No results")
    else:

        ############################
        # compile scores
        ############################
        # calculate accuracy
        accuracy = sum(results) / len(results)
        print_rank_0(f"Results: {accuracy}, on {len(results)} prompts")

        # compute mean and std of scores, chosen and rejected, then margin between them
        print_rank_0(f"Mean chosen: {np.mean(scores_chosen)}, std: {np.std(scores_chosen)}")
        print_rank_0(f"Mean rejected: {np.mean(scores_rejected)}, std: {np.std(scores_rejected)}")
        print_rank_0(f"Mean margin: {np.mean(np.array(scores_chosen) - np.array(scores_rejected))}")

        out_dataset = dataset.add_column("results", results)
        out_dataset = out_dataset.add_column("subsets", subsets)
        out_dataset = out_dataset.to_pandas()  # I know this is meh

        results_grouped = {}
        present_subsets = np.unique(out_dataset["subsets"])
        for subset in present_subsets:
            subset_dataset = out_dataset[out_dataset["subsets"] == subset]
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            print_rank_0(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

        results_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print_rank_0(f"Results: {results_section}")

if __name__ == "__main__":
    with torch.no_grad():
        main()
