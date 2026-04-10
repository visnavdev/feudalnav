# FeudalNav

To create the human click training data, run 

```python deterministicTrials/makeData_detWork_HumMid_detHigh.py```

Don't forget to change the paths (either using the arg flags on directly in the code) to the dataset files and code.

To create data for all environments starting with a certain letter, run 

```python deterministicTrials/makeData_detWork_HumMid_detHigh.py --letter a``` 

(For example, this works to run all envs starting with a). 
