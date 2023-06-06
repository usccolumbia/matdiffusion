## MatDiffusion, diffusion models for material composition design

Rongzhi Dong, Nihang Fu, Jianjun Hu, Edirisuriya M. D. Siriwardane  
Machine Learning and Evolution Laboratory  
Department of Computer Science and Engineering  
University of South Carolina

```bibtex
@article{dong2023matdiff,
  title={Generative Design of inorganic compounds using deep diffusion language models},
  author={Rongzhi Dong, Nihang Fu, Jianjun Hu, Edirisuriya M. D. Siriwardane},
  journal={arXiv preprint arXiv:xxxxx},
  year={2023}
}

```


### Installation and use Diffusion-LM

1) Install Diffusion-LM

~~~Conda Setup:
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb 
~~~

2) Train Diffusion-LM:


```cd Diffusion-LM/improved-diffusion 
mkdir diffusion_models
sh run.sh
```

the trained model is saved in ./diffusion_models

3) Sample from Diffusion-LM:

```
mkdir generation_outputs
sh decode.sh
```

The generation is saved in ./generation_outputs.

```
sq2formula.py
```

The sequences are then conver to formulas and the formula results are saved to formulas.csv

### Installation and use Diffusion-BERT
1) Install Diffusion-BERT

```Conda Setup:
conda create --name DB python=3.8
conda activate DB
pip install -r requirements.txt
```

2) Train Diffusion-BERT:

```
cd Diffusion-BERT
python word_freq.py 
```

to get the frequency in the text corpus

```
sh run.sh
```
for unconditional generation

3) Sampling from Diffusion-BERT:

Pass the path to the checkpoint obtained during training to predict.py to unconditionally sample from DiffusionBERT.

The generated sequences are saved to temp.txt

```
python sq2formula.py
```
The sequences are then conver to formulas and the formula results are saved to formulas.csv

## Acknowledgement 
Our work is based on two text-generation diffusion models including 
[Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/pdf/2205.14217.pdf) and [DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://arxiv.org/pdf/2211.15029.pdf) 



