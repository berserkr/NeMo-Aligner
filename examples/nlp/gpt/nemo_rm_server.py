import torch

from pytorch_lightning.trainer.trainer import Trainer
from typing import List

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.utils.text_generation_utils import tokenize_batch
from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, set_autocast_gpu_dtype

from nemo_aligner.utils.distributed import run_distributed_inference
from nemo_aligner.utils.server_utils import (
    calculate_inference_batch_padding_multiple,
    pad_batch_and_strip_sequence
)

from utils.rpc import RPCServer

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

GLOBAL_CONFIG = dict()

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

    """

    assert len(data) == 2

    user_data = data[0]
    assistant_data = data[1]

    batch_size = len(user_data['content'])

    user_content = user_data['content']
    assistant_content = assistant_data['content']

    samples = []
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

    samples = dataloader_to_template(sample_batch)

    reward_batch = get_reward(
        samples, infer_fn=infer_fn, tokenize_func=tokenize_func, model_forward_micro_batch_size=model_forward_micro_batch_size, strip_sequence_length_to_multiple=strip_sequence_length_to_multiple
    )

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

    return labels, scores


def get_rm_scores(sentence_batch):
    """
    Takes a string and returns a raw score. String to be formatted by user.

    Returns a dict of labels and scores
    """
    infer_fn = GLOBAL_CONFIG['infer_fn']
    tokenize_func = GLOBAL_CONFIG['tokenize_func']
    model_forward_micro_batch_size = GLOBAL_CONFIG['model_forward_micro_batch_size']
    strip_sequence_length_to_multiple = GLOBAL_CONFIG['strip_sequence_length_to_multiple']

    reward_batch = get_reward(
        sentence_batch, 
        infer_fn=infer_fn, 
        tokenize_func=tokenize_func, 
        model_forward_micro_batch_size=model_forward_micro_batch_size, 
        strip_sequence_length_to_multiple=strip_sequence_length_to_multiple
    )

    reward_out = dict()
    reward_out['labels'] = []
    reward_out['scores'] = []

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

        reward_each = [min(4.0, max(0.0, float(r))) for r in reward]
        reward_each = [round(r) for r in reward_each]

        if valid_5:
            reward_string = ",".join(f"{a}:{r}" for a, r in zip(HELPSTEER_ATTRIBUTES, reward_each))
            score = sum([REWARD_VECTOR[4:][i] * reward[i] for i in range(0,len(REWARD_VECTOR[4:]))])
        else:
            reward_string = ",".join(f"{a}:{r}" for a, r in zip(ALL_STEERLM_ATTRIBUTES, reward_each))
            score = sum([REWARD_VECTOR[i] * reward[i] for i in range(0,len(REWARD_VECTOR))])
        
        reward_out['labels'] .append(reward_string)
        reward_out['scores'].append(score)

    return reward_out


@hydra_runner(config_path="conf", config_name="inference_rm")
def main(cfg) -> None:

    # Load the model
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

    model_forward_micro_batch_size = cfg.model.get("forward_micro_batch_size", cfg.model.micro_batch_size)
    strip_sequence_length_to_multiple = cfg.inference.get("strip_sequence_length_to_multiple", None)

    GLOBAL_CONFIG['infer_fn'] = infer_fn
    GLOBAL_CONFIG['tokenize_func'] = tokenize_func
    GLOBAL_CONFIG['model_forward_micro_batch_size'] = model_forward_micro_batch_size
    GLOBAL_CONFIG['strip_sequence_length_to_multiple'] = strip_sequence_length_to_multiple

    port = cfg.inference.get("port", 7777) #must be an int
    host = cfg.inference.get("host", '0.0.0.0') #must be a str
 
    server = RPCServer(host=host, port=port)

    server.registerMethod(get_rm_scores)

    server.run()

if __name__ == "__main__":
    with torch.no_grad():
        main()
