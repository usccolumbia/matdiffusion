TOKENIZERS_PARALLELISM=false, CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 1 --master_port 29503 main.py \
  --lr 5e-5 \
  --batch_size 128 \
  --timestep 'layerwise' \
  --from_scratch false
