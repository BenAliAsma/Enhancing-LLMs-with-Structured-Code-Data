# Comparison with canceling sub-expression
If a comparison contains multiple linear sub-expressions which cancel each other out, then the comparison can be simplified by removing them. This may be indicative of a coding error.


## Recommendation
Check that the redundant sub-expressions are not the result of a coding error. If not, then simplify the comparison by removing them.


## Example

```cpp
bool compare_xyz(unsigned short x, unsigned short y, unsigned short z) {
  return (x + y < x + z);  // x can be canceled
}

```
