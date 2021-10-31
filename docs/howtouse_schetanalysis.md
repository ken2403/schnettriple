## How to use spkanal module

##### *class*  **schnetanalysis.SchnetAnalysis**

```python
from schnetanalysis import SchnetAnalysis as sa

# construct the SchnetAnalysis object
analysis = sa(modeldir, dbasepath)
```
- Attributes <br>
    - *modeldir* : str or path-like object<br>
        Specify the directory where the model information learned by schnet is stored.<br>
    - *databasepath* : str or path-like object<br>
        Specify the path of the database to use.<br>
    - *available\_properties*<br>
        Display computable properties as a list.<br>

- Methods <br>
    - *log\_plot(error='RMSE', props=['energy', 'forces'], units=['eV', 'eV/\u212B'], axes=None, verbose=True)*
    - *inout\_property(prop='energy', data='train', divided_by_atoms=True, device='cpu', start=0, stop=None, save=True, \_return=False)*
    - *inout\_plot(prop='energy', data=['train', 'test'], axes=None, line=True, xlabel='DFT energy', unit='eV/atom')*
    - *rmse(prop='energy', data='train', save=True)*