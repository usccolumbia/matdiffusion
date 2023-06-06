from pymatgen.core.composition import Composition
import pandas as pd

filename='temp.txt'

sq=open(filename,'r')
lines=sq.readlines()

formulas =[]
# ## Diffusion LM
# for line in lines:
#     if ' END START ' not in line:
#         continue
#     if '.' not in line:
#         continue
# #     print(line)
#     line = line.strip()
#     line = line[11:-1]
#     # print(line)
#     try:
#         formula = Composition(line.replace(' ', '')).to_pretty_string()
#     except:
#         print('error in ',line)
#     formulas.append(formula)

## Diffusion BERT
for line in lines:
    if '.' not in line:
        continue
    # print(line)
    # line = line.strip()
    # print(line)
    end = line.index('.')
    line = line[:end]
    try:
        formula = Composition(line.replace(' ', '')).to_pretty_string()
    except:
        print('error in ',line)
    formulas.append(formula)
    formulas = list(set(formulas))
df = pd.DataFrame(formulas)

df.to_csv('formulas.csv', index=False, header=['pretty_formula'])


