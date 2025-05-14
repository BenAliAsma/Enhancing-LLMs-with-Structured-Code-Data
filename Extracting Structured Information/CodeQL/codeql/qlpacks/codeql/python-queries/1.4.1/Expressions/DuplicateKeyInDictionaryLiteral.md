# Duplicate key in dict literal
Dictionary literals are constructed in the order given in the source. This means that if a key is duplicated the second key-value pair will overwrite the first as a dictionary can only have one value per key.


## Recommendation
Check for typos to ensure that the keys are supposed to be the same. If they are then decide which value is wanted and delete the other one.


## Example
This example will output "c" because the mapping between 2 and "b" is overwritten by the mapping from 2 to "c". The programmer may have meant to map 3 to "c" instead.


```python
dictionary = {1:"a", 2:"b", 2:"c"}
print dictionary[2]
```

## References
* Python: [Dictionary literals](http://docs.python.org/2/reference/expressions.html#dictionary-displays).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
