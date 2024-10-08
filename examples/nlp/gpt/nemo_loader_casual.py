import torch

from pytorch_lightning.trainer.trainer import Trainer

from nemo.core.config import hydra_runner
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.modules.common.lm_utils import pad_batch

from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, set_autocast_gpu_dtype
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor
from nemo_aligner.utils.server_utils import (
    calculate_inference_batch_padding_multiple,
    pad_batch_and_strip_sequence
)

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def run_distributed_generate(inputs=None, length_params=None, sampling_params=None, generate_fn=None):
    tokens, lengths = None, None
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    is_rank_0 = torch.distributed.get_rank() == 0

    if is_rank_0:
        tokens = torch.as_tensor(inputs["inputs"], dtype=torch.long, device=torch.cuda.current_device())
        lengths = torch.as_tensor(inputs["sequence_length"], dtype=torch.long, device=torch.cuda.current_device())

    tokens = broadcast_2d_tensor(tokens, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank]
    lengths = broadcast_2d_tensor(lengths, 0, dtype=torch.long, group=None).chunk(dp_size)[dp_rank].squeeze(-1)

    outputs = generate_fn(inputs=(tokens, lengths), length_params=length_params, sampling_params=sampling_params)

    return outputs 


def generate(
        generate_fn, 
        tokenizer, 
        model_forward_micro_batch_size, 
        strip_sequence_length_to_multiple, 
        length_params, 
        max_len,
        sampling_params,
        sentences
    ):

    def tokenize(sentence):
        tokens = tokenizer.encode_chat_completion(sentence).tokens
        return tokens

    context_tokens = list(map(tokenize, sentences))
    max_sequence_length = max(len(x) for x in context_tokens)

    context_tokens, context_lengths = pad_batch(
        context_tokens, 
        tokenizer.instruct_tokenizer.tokenizer.eos_id, 
        max_len - max_sequence_length)
    
    context_tokens = [x[:max_len] for x in context_tokens]
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    tokens, sequence_lengths = context_tokens_tensor, context_length_tensor
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

    outputs = run_distributed_generate(
        inputs=inputs, 
        generate_fn=generate_fn, 
        length_params=length_params,
        sampling_params=sampling_params)

    return outputs


@hydra_runner(config_path="conf", config_name="inference_mistral_instruct")
def main(cfg) -> None:

    # Load the model
    cfg.model = load_and_override_model_config(cfg.model_file, cfg.model)
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    set_autocast_gpu_dtype(cfg.trainer.precision)
    if trainer.precision == "16":
        cfg.model.megatron_amp_O2 = False
    elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get("megatron_amp_O2", False):
        cfg.model.megatron_amp_O2 = True

    ptl_model = MegatronGPTModel.restore_from(
        cfg.model_file, trainer=trainer, override_config_path=cfg.model
    )

    ptl_model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        ptl_model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    
    # initialize the model for inferencing
    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    ptl_model = ptl_model.cuda()
    model_forward_micro_batch_size = cfg.model.get("forward_micro_batch_size", cfg.model.micro_batch_size)
    strip_sequence_length_to_multiple = cfg.inference.get("strip_sequence_length_to_multiple", None)

    # use v3 tokenizer for instruct model
    tokenizer = MistralTokenizer.v3(is_tekken=True)

    prompt = "How expensive would it be to ask a window cleaner to clean all windows in Paris. Make a reasonable guess in US Dollar."
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

    sample_batch_size = 8
    sentences = [completion_request] * sample_batch_size

    max_len = 4096 # seq len

    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
        "end_strings": cfg.inference.end_strings,
    }

    outputs = generate(
        generate_fn=ptl_model.generate, 
        tokenizer=tokenizer, 
        model_forward_micro_batch_size=model_forward_micro_batch_size,
        strip_sequence_length_to_multiple=strip_sequence_length_to_multiple,
        length_params=length_params,
        max_len=max_len,
        sampling_params=sampling_params,
        sentences=sentences
        )    
    
    print_rank_0(outputs['sentences'])


if __name__ == "__main__":
    with torch.no_grad():
        main()
