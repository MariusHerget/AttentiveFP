## Conda install

```
conda create -n <env-name>
conda activate <env-name>
conda install --yes cuda -c nvidia/label/cuda-12.4.0
conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install --yes scikit-learn rdkit seaborn matplotlib tensorboardX
conda install --yes jupyter jupytext
jupyter notebook --no-browser
```
