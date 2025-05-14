# Missing return-value check for a 'scanf'-like function
This query finds calls of `scanf`-like functions with missing or improper return-value checking.

Specifically, the query flags uses of variables that may have been modified by `scanf` and subsequently are used without being guarded by a correct return-value check. A proper check is one that ensures that the corresponding `scanf` has returned (at least) a certain minimum constant.

Functions in the `scanf` family return either EOF (a negative value) in case of IO failure, or the number of items successfully read from the input. Consequently, a simple check that the return value is truthy (nonzero) is not enough.

> WARNING: This query has medium precision because, in the current implementation, it takes a strict stance on unguarded uses of output variables, and flags them as problematic even if they have already been initialized.

## Recommendation
Ensure that all subsequent uses of `scanf` output arguments occur in a branch of an `if` statement (or similar), in which it is known that the corresponding `scanf` call has in fact read all possible items from its input. This can be done by comparing the return value to a numerical constant.


## Example
This example shows different ways of guarding a `scanf` output:


```cpp
{
  int i, j, r;

  r = scanf("%d %d", &i, &j);

  use(i); // BAD: i is not guarded

  if (r >= 1) {
    use(i); // GOOD: i is guarded correctly
    use(j); // BAD: j is guarded incorrectly
  }

  if (r != 2)
    return;

  use(j); // GOOD: j is guarded correctly
}

```

## References
* SEI CERT C++ Coding Standard: [ERR62-CPP. Detect errors when converting a string to a number](https://wiki.sei.cmu.edu/confluence/display/cplusplus/ERR62-CPP.+Detect+errors+when+converting+a+string+to+a+number).
* SEI CERT C Coding Standard: [ERR33-C. Detect and handle standard library errors](https://wiki.sei.cmu.edu/confluence/display/c/ERR33-C.+Detect+and+handle+standard+library+errors).
* cppreference.com: [scanf, fscanf, sscanf, scanf_s, fscanf_s, sscanf_s](https://en.cppreference.com/w/c/io/fscanf).
* Common Weakness Enumeration: [CWE-252](https://cwe.mitre.org/data/definitions/252.html).
* Common Weakness Enumeration: [CWE-253](https://cwe.mitre.org/data/definitions/253.html).
