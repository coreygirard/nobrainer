# nobrainer



```python
data = [[[0, 0, 1],0],
        [[0, 1, 1],1],
        [[1, 0, 1],1],
        [[0, 1, 0],1],
        [[1, 0, 0],1],
        [[1, 1, 1],0],
        [[0, 0, 0],0]]

nn = DeepNetwork(data,2,4)
nn.train(60000)

print(nn.think([1, 1, 0]))
```
