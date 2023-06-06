from pymatgen.core.composition import Composition
import pandas as pd

filename='./generation_outputs_resnet6/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s6_d0.1_sd102_xstart_e2e.ema_0.9999_200000.pt.samples_-1.0.txt'\

sq=open(filename,'r')
lines=sq.readlines()

formulas =[]
for line in lines:
    if ' END START ' not in line:
        continue
    if '.' not in line:
        continue
#     print(line)
    line = line.strip()
    line = line[11:-1]
#     print(line)
    try: 
        formula = Composition(line.replace(' ', '')).to_pretty_string()
    except:
        print('error in ',line)
    formulas.append(formula)
    formulas = list(set(formulas))
df = pd.DataFrame(formulas)

df.to_csv('formulas.csv', index=False, header=['pretty_formula'])


