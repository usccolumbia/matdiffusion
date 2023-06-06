# matdiffusion
Diffusion model for material composition design
﻿Diffusion-LM Improves Controllable Text Generation
https://arxiv.org/pdf/2205.14217.pdf 

Conda Setup:
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb

Train Diffusion-LM:
cd improved-diffusion; mkdir diffusion_models;
Run run.sh, the trained model is saved in ./diffusion_models

Sample from Diffusion-LM:
mkdir generation_outputs 
Run 'decode.sh', the generation is saved in ./generation_outputs
Run sq2formula.py to get the formula based on the sequences, the formula results are saved to formulas.csv


DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models
https://arxiv.org/pdf/2211.15029.pdf 

Conda Setup:
conda create --name DB python=3.8
conda activate DB
pip install -r requirements.txt

Train Diffusion-BERT:
python word_freq.py to get the frequency in the text corpus
'run.sh' for unconditional generation

Sampling from Diffusion-BERT:
Pass the path to the checkpoint obtained during training to predict.py to unconditionally sample from DiffusionBERT.
The generated sequences are saved to temp.txt
Run sq2formula.py to get the formula based on the sequences, the formula results are saved to formulas.csv
