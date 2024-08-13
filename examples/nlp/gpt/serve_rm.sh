#!/bin/bash
export PYTHONPATH=/opt/NeMo-Framework-Launcher/launcher_scripts/:/opt/NeMo/:/opt/NeMo-Aligner/:/opt/Megatron-LM/

python /opt/NeMo-Aligner/examples/nlp/gpt/serve_reward_model.py \
	rm_model_file=/data/models/nue7b_arm-20240801T013647 \
	trainer.num_nodes=1 \
	trainer.devices=2 \
	++model.tensor_model_parallel_size=1 \
	++model.pipeline_model_parallel_size=1 \
	inference.inference_micro_batch_size=2 \
	inference.port=8989
