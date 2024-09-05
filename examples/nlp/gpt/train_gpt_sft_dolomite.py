# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#dolomite stuff
import torch
import copy

from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, TrainingArguments
from accelerate import Accelerator

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import dolomite_engine.hf_models

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    resolve_and_create_trainer,
)

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

"""Script to start SFT training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


PREFIX_STR = (
    "\x00"  # the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
)

IGNORE_INDEX = -100
SYSTEM_TOKEN = "System"

TYPE_INSTRUCTION = {
    'TEXT_TO_VALUE': "",
    'VALUE_TO_TEXT': '',
}

SPECIAL_TOKENS = {
    'system_turn_start' : "\x00",
    'turn_start' : "\x11",
    'label_start' : "\x12",
    'end_of_turn' : "\x0A",
    'end_of_name': "\x0A"
}

def _get_header_conversation_type_mask_role(source, special_tokens):
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    data_type = None
    if 'type' in source:
        data_type = source['type']
        if data_type is not None:
            assert data_type in TYPE_INSTRUCTION, f"source type {data_type} not supported"
    # add end signal and concatenate together
    conversation = source['system']
    if data_type is not None:
        if TYPE_INSTRUCTION[data_type] != '':
            conversation = conversation + '\n' + TYPE_INSTRUCTION[data_type]
    mask_role = source.get('mask', 'User')
    header = f"{special_tokens['system_turn_start']}{SYSTEM_TOKEN}{END_NAME_SIGNAL}{conversation}{END_SIGNAL}"
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)
    return header, conversation, data_type, mask_role


def get_prompt_template_example(special_tokens):
    source = {
        'system': '{system message}',
        'conversations': [
            {'from': 'User', 'value': '{turn 1 user message}', 'label': None},
            {'from': 'Assistant', 'value': '{turn 1 assistant message}', 'label': '{turn 1 assistant label}'},
            {'from': 'User', 'value': '{turn 2 user message}', 'label': None},
            {'from': 'Assistant', 'value': '{turn 2 assistant message}', 'label': '{turn 2 assistant label}'},
        ],
        "mask": "User",
        "type": "VALUE_TO_TEXT",
    }
    _, conversation, _, _ = _get_header_conversation_type_mask_role(source, special_tokens)
    return conversation


def identify_start_index_of_subsequence(subsequence, sequence):
    """ find the location of the small tensor in the large tensor.
        e.g.  small = [1,3], large = [2,3,1,3], returns 2
              small = [3,2], large = [2,3,1,3], returns -1
    Args:
        small (tensor): small tensor
        large (tensor): large tensor
    """
    for i in range(sequence.size(0) - subsequence.size(0) + 1):
        if torch.equal(sequence[i : i + subsequence.size(0)], subsequence):
            return i
    return -1


def _mask_targets(
    target,
    tokenized_lens,
    speakers,
    header_len,
    s_ids,
    tokenizer,
    mask_role,
    gtype,
    name_end_token_ids,
    special_tokens,
    label_start_ids,
    num_turn_start_tokens,
):
    """ This function masks the tokens so the loss is computed only on the non-masked role's responses.
    For 'TEXT_TO_VALUE' type, the loss is computed on the value attributes.

    Args:
        target (Tensor): input ids
        tokenized_lens (List[int]): array of lengths of each turns
        speakers (List[str]): array of speakers of each turns
        header_len (int): the system prompt length
        s_ids (List[Tensor]): array of tokenized ids of each turns
        tokenizer (TokenizerSpec): tokenizer object
        mask_role (str): the speaker id to be masked from loss computation
        gtype (str): either 'TEXT_TO_VALUE' or 'VALUE_TO_TEXT'
        name_end_token_ids (int): end of name token ids
        special_tokens (dict): special tokens used for the chat prompt. It has the keys: system_turn_start, turn_start, label_start, end_of_turn
        label_start_ids (list): list of label start token ids,
        num_turn_start_tokens (int): number of tokens of the turn_start str
    """
    TURN_TOKEN = special_tokens['turn_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    label_start_ids = torch.tensor(label_start_ids)
    name_end_token_ids = torch.tensor(name_end_token_ids)

    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id) in enumerate(zip(tokenized_lens, speakers, s_ids)):
        # note, sentence piece will add extra empty token in front. has to compute the diff
        id1 = tokenizer.text_to_ids(PREFIX_STR)
        id2 = tokenizer.text_to_ids(PREFIX_STR + TURN_TOKEN + speaker + END_NAME_SIGNAL)
        skip_name_len = len(id2) - len(
            id1
        )  # s_ids[:skip_name_len] is the name part of the prompt 'TURN_TOKEN + speaker + END_NAME_SIGNAL'
        # get the position of the label start string in this turn
        location = identify_start_index_of_subsequence(label_start_ids, s_id)

        if location >= 0:
            # if it contains the label start tokens
            if gtype == 'VALUE_TO_TEXT':
                # handles the case that condition on labels to generate respone
                # the next token after the name part of the prompt is the beginning of the label start tokens
                assert skip_name_len == location
                # find the first new line token after the label part, which indicates the end of the whole label string
                # newline_loc = torch.where((s_id[skip_name_len:] == name_end_token_ids))[0]
                newline_loc = identify_start_index_of_subsequence(name_end_token_ids, s_id[skip_name_len:])
                if newline_loc < 0:
                    # cannot find new line token, which means the the whole turn is just a partial label string. Mask the whole turn
                    target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
                    continue
                # skip the label part and the new line token
                more_skip_len = newline_loc + len(name_end_token_ids)
                # skip the name part and the label part
                skip_name_len += more_skip_len
            elif gtype == 'TEXT_TO_VALUE':
                # handles the case that condition on response to generate label
                # skip the name part, response and the label start tokens part, the remainder is the label string without label start, e.g. 'quality:9,toxicity:8...'
                skip_name_len = location + len(label_start_ids)
        if cur_idx >= tgt_len:
            break
        elif cur_idx + tokenized_len < tgt_len:
            # Check whether the mask is applied to the correct position, the first token is turn start tokens
            if not torch.equal(target[cur_idx + 1 : cur_idx + tokenized_len], s_id[1:]):
                logging.warning("a sentence mismatches the corresponding piece " "in the conversation")
        if i == 0 and (gtype == 'VALUE_TO_TEXT' or gtype is None):
            # mask the first turn completely to provide at least one turn as context for the rest
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and i == 1 and gtype == 'TEXT_TO_VALUE':
            # leave the first turn start tag unmasked, servers severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i > 1):
            # leave the first turn start tag unmasked, which severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i <= 1):
            # mask out everything in the second turn
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            # mask up to name part, label part for VALUE_TO_TEXT, or name part, response and label start tokens for TEXT_TO_VALUE, or just the name part if gtype is None
            target[cur_idx : cur_idx + skip_name_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def response_value_formater(label, label_start, end_signal):
    if isinstance(label, str):
        return label_start + label + end_signal
    elif label is None:
        return ''
    else:
        raise ValueError(f'Unknown label type {type(label)}, only str type is supported')


def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
    TURN_TOKEN = special_tokens['turn_start']
    END_SIGNAL = special_tokens['end_of_turn']
    LABEL_START = special_tokens['label_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = ""
    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]
        role_token = TURN_TOKEN
        if gtype is None:
            sentence["value"] = (
                BEGIN_SIGNAL + role_token + sentence_from + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
            )
        elif gtype == "VALUE_TO_TEXT":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
                + sentence["value"]
                + END_SIGNAL
            )
        elif gtype == "TEXT_TO_VALUE":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + sentence["value"]
                + END_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
            )
        else:
            raise ValueError(
                f"source type {gtype} not supported, only 'VALUE_TO_TEXT' and 'TEXT_TO_VALUE' are supported"
            )
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end, which will be included for loss calculation
        if sentence_from != mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN
    return conversation


def formatting_prompts_func(samples):
    """ Takes samples in the format:
    {'conversations': [[{'value': 'c#', 'from': 'User', 'label': None},
    {'value': 'C# is a high-level, object-oriented programming language developed by Microsoft as part of its .NET initiative. It was created as a modern alternative to Java and supports a variety of programming paradigms, including imperative, functional, and event-driven. C# is primarily used for Windows application development, but it can also be used for web, mobile, and game development. The language is designed to be safe, secure, and efficient, and it provides developers with a rich set of libraries and tools for building robust and scalable applications. C# is also widely used in the game development industry, particularly in the development of games for the Xbox 360 and Xbox One consoles.',
        'from': 'Assistant',
        'label': 'helpfulness:3,correctness:4,coherence:4,complexity:2,verbosity:2'}],
    [{'value': 'c#', 'from': 'User', 'label': None},
    {'value': 'C# (pronounced "C sharp") is a modern, object-oriented programming language developed by Microsoft. It is widely used for building various types of applications, including web applications, desktop applications, mobile applications, and games. C# is similar to other programming languages such as Java and C++, and it is known for its simplicity and ease of use. C# is a powerful language that provides a rich set of libraries and frameworks that make it easy to build robust and scalable applications.\n\nHere is a brief overview of some key features of C#:\n\n1. Object-oriented: C# is an object-oriented language, which means it uses the concept of objects to represent real-world entities and their behavior.\n\n2. Cross-platform: C# can be used to build applications for multiple platforms, including Windows, macOS, and Linux.\n\n3. Strongly typed: C# is a strongly typed language, which means that variables must be declared with a specific type, and their type cannot be changed at runtime.\n\n4. Event-driven: C# uses an event-driven programming model, which means that programs are built around the concept of events, such as user input or network activity.\n\n5. Garbage-collected: C# has a garbage collector that automatically manages memory allocation and deallocation, making it easier to write memory-efficient and robust applications.\n\n6. Community-driven: C# has a large and active community of developers, who contribute to the language and its libraries through open-source projects and other initiatives.\n\nOverall, C# is a versatile and powerful programming language that is widely used for building a variety of applications.',
        'from': 'Assistant',
        'label': 'helpfulness:3,correctness:4,coherence:4,complexity:3,verbosity:3'}],
    [{'value': 'bacillus subtilus', 'from': 'User', 'label': None},
    {'value': 'Bacillus subtilis is a Gram-positive, rod-shaped bacterium commonly found in soil and water. It is a facultative anaerobe that can grow in the presence or absence of oxygen. B. subtilis is a model organism for studying bacterial cell biology, genetics, and physiology, and has many industrial and medical applications.\n\nB. subtilis is a versatile organism with many interesting properties. It can form spores that allow it to survive harsh environmental conditions, such as heat, cold, and dehydration. The spores can remain dormant for long periods of time and then germinate when conditions are favorable. B. subtilis can also produce a variety of enzymes, antibiotics, and other compounds that have industrial and medical applications.\n\nB. subtilis has been used in the production of enzymes for industrial applications, such as in the food and beverage industry for the production of cheese, yogurt, and beer. It is also used in the production of antibiotics and other pharmaceutical compounds. B. subtilis is also a model organism for studying bacterial cell biology, genetics, and physiology, and has been used extensively in research to study the mechanisms of bacterial growth, development, and survival.\n\nOverall, B. subtilis is a versatile and important organism with many industrial and medical applications. Its properties and behaviors make it a valuable tool for researchers and industry alike.',
        'from': 'Assistant',
        'label': 'helpfulness:3,correctness:4,coherence:4,complexity:3,verbosity:3'}],
    [{'value': 'bacillus subtilus', 'from': 'User', 'label': None},
    {'value': 'The Bacillus subtilis is a Gram-positive, endospore-forming bacterium that is commonly found in the environment, such as in soil and on plants. It is a rod-shaped bacterium that can grow in a wide range of temperatures and pH levels.\n\nBacillus subtilis is a common bacterium used in the food and beverage industry as a starter culture for the production of fermented foods, such as yogurt and sauerkraut. It is also used in the production of various enzymes, such as amylase and protease, which are used in the baking, brewing, and detergent industries.\n\nBacillus subtilis has also been studied for its potential use in biotechnology and medicine. It has been shown to have antimicrobial activity against some pathogens and is being studied for its potential use in the treatment of infections and other diseases. Additionally, some strains of Bacillus subtilis have been shown to produce antibiotics, such as bacitracin, which can be used to treat bacterial infections.\n\nOverall, Bacillus subtilis is a versatile and important bacterium that has a wide range of applications in various industries and fields of study.',
        'from': 'Assistant',
        'label': 'helpfulness:3,correctness:4,coherence:4,complexity:2,verbosity:2'}]],
    'system': ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."],
    'mask': ['User', 'User', 'User', 'User'],
    'type': ['VALUE_TO_TEXT', 'VALUE_TO_TEXT', 'VALUE_TO_TEXT', 'VALUE_TO_TEXT']}
    """
    output_texts = []

    # iterate over batch
    for i in range(0, len(samples['conversations'])):
        sample = dict()
        for k in samples.keys():
            sample[k] = samples[k][i]
        _, conversation, _, _ = _get_header_conversation_type_mask_role(sample, SPECIAL_TOKENS)
        output_texts.append(conversation)
    return output_texts

class DolomiteTokenizer(GPT2TokenizerFast):

    # new functionality
    def text_to_ids(self, text):
        return self(text)["input_ids"]


@hydra_runner(config_path="conf", config_name="gpt_sft")
def main(cfg) -> None:
    global SPECIAL_TOKENS

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    # load dolomite model
    model = AutoModelForCausalLM.from_pretrained(cfg.model.restore_from_path, device_map={"": Accelerator().process_index})
    print(model)

    tokenizer = DolomiteTokenizer.from_pretrained(cfg.model.restore_from_path)

    # these changes cause aten errors due to mismatches
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # pad token = eos token leads to bad behavior
    #tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    train_data_cfg = cfg.model.data.train_ds
    val_data_cfg = cfg.model.data.validation_ds

    """
    if cfg.model.data.get("sample", False):
        # if it is negative, num_samples is None
        if cfg.trainer.sft.max_steps < 0:
            num_samples = None
        else:
            num_samples = cfg.trainer.sft.max_steps * train_data_cfg.global_batch_size
    else:
        num_samples = None
    """

    train_ds = load_dataset('json', data_files=train_data_cfg.file_path, split='train')
    val_ds = load_dataset('json', data_files=val_data_cfg.file_path, split='train')

    # special tokens
    SPECIAL_TOKENS['system_turn_start'] = cfg.model.data.chat_prompt_tokens.system_turn_start
    SPECIAL_TOKENS['turn_start'] = cfg.model.data.chat_prompt_tokens.turn_start
    SPECIAL_TOKENS['label_start'] = cfg.model.data.chat_prompt_tokens.label_start

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    extra_kwargs = {"min_lr": cfg.model.optim.sched.min_lr}
    training_args = TrainingArguments(
        output_dir=cfg.cpts,
        dataloader_drop_last=True,
        num_train_epochs=cfg.trainer.sft.max_epochs,
        evaluation_strategy="steps",
        eval_steps=cfg.trainer.sft.val_check_interval,
        save_steps=cfg.trainer.sft.save_interval,
        save_total_limit=5,
        logging_steps=1,
        per_device_train_batch_size=train_data_cfg.global_batch_size,
        per_device_eval_batch_size=val_data_cfg.global_batch_size,
        learning_rate=cfg.model.optim.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs=extra_kwargs,
        optim="paged_adamw_32bit",
        warmup_steps=cfg.model.optim.sched.warmup_steps,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        fp16=False,
        bf16=True,
        weight_decay=cfg.model.optim.weight_decay,
        run_name="dolomite_sft_steerlm",
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0
    )

    response_template = "<extra_id_1>Assistant\n<extra_id_2>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.encoder_seq_length,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()
    trainer.model.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    main()
