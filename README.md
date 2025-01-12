# 人工智能中的编程 大作业
吴童 2200013212

### Codebase
- `csrc/`: CUDA codes 
- `mytorch/`: PyTorch-like AI framework
- `Makefile`: Convenient scripts

## Usage
### Environment Configuration
```sh
conda install -f env.yaml
```

### Install
```sh
make
```
or 
```sh
python setup.py install 
```
### Test
```sh
make test
```
or
```sh
cd tests
pytest test_xxx.py
```
### Import
```python
import mytorch
from mytorch import Tensor, nn, datasets
```

## Extra Info
- Python Version: 3.12.3
- CUDA Version: 12.4

