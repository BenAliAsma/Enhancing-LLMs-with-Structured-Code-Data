# Dubious NULL check
The expression `&foo->bar` gets the address of `foo`'s member `bar`, which is the address of `foo` plus the offset of the `bar` member. If said offset is non-zero, then the expression `&foo->bar` only equals `NULL` when the address of `foo` is negative. While this is not impossible, it can only happen if `foo` is a negative integer explicitly cast to a pointer, or if `foo` is a pointer into kernel-mode address space. As neither of these cases are particularly likely, the `NULL`-check is dubious.


## Recommendation
Either the `NULL`-check is entirely redundant, or the wrong thing is being checked against `NULL`. In the former case, the check can be replaced with boolean `true` or `false`, and then the surrounding context can be simplified. In the latter case, consider which sub-expressions might be `NULL`, and test them instead. In particular, simply removing the ampersand may yield a more suitable expression to test.


## Example
```cpp

struct person {
  int id;
  char* name;
};

bool hasName(person* p) {
  return  p       != NULL  // This check is sensible,
      &&  p->name != NULL  // as is this one.
      && &p->name != NULL; // But this check is dubious.
}

```
