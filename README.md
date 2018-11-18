# nobrainer

[![Build Status](https://travis-ci.org/coreygirard/nobrainer.svg?branch=master)](https://travis-ci.org/coreygirard/nobrainer) <br>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

```python
data = [[[0, 0, 1],0],
        [[0, 1, 1],1],
        [[1, 0, 1],1],
        [[0, 1, 0],1],
        [[1, 0, 0],1],
        [[1, 1, 1],0],
        [[0, 0, 0],0]]

nn = nobrainer.DeepNetwork(data,2,4)
nn.train(60000)

print(nn.think([1, 1, 0]))
```

```python
[ 0.0078876]
```
Heavily inspired by Milo Spencer-Harper's excellent article [here](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1#.vncojtrlw).
