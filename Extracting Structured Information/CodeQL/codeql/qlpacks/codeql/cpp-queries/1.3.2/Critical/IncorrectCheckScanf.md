# Incorrect return-value check for a 'scanf'-like function
This query finds calls of `scanf`-like functions with improper return-value checking. Specifically, it flags uses of `scanf` where the return value is only checked against zero.

Functions in the `scanf` family return either `EOF` (a negative value) in case of IO failure, or the number of items successfully read from the input. Consequently, a simple check that the return value is nonzero is not enough.


## Recommendation
Ensure that all uses of `scanf` check the return value against the expected number of arguments rather than just against zero.


## Example
The following examples show different ways of guarding a `scanf` output. In the BAD examples, the results are only checked against zero. In the GOOD examples, the results are checked against the expected number of matches instead.


```cpp
{
  int i, j;

  // BAD: The result is only checked against zero
  if (scanf("%d %d", &i, &j)) { 
      use(i);
      use(j);
  }

  // BAD: The result is only checked against zero
  if (scanf("%d %d", &i, &j) == 0) { 
    i = 0;
    j = 0;
  }
  use(i);
  use(j);

  if (scanf("%d %d", &i, &j) == 2) { 
      // GOOD: the result is checked against 2
  }

  // GOOD: the result is compared directly
  int r = scanf("%d %d", &i, &j);
  if (r < 2) {
    return;
  }
  if (r == 1) { 
    j = 0;
  }
}

```

## References
* SEI CERT C++ Coding Standard: [ERR62-CPP. Detect errors when converting a string to a number](https://wiki.sei.cmu.edu/confluence/display/cplusplus/ERR62-CPP.+Detect+errors+when+converting+a+string+to+a+number).
* SEI CERT C Coding Standard: [ERR33-C. Detect and handle standard library errors](https://wiki.sei.cmu.edu/confluence/display/c/ERR33-C.+Detect+and+handle+standard+library+errors).
* cppreference.com: [scanf, fscanf, sscanf, scanf_s, fscanf_s, sscanf_s](https://en.cppreference.com/w/c/io/fscanf).
* Common Weakness Enumeration: [CWE-253](https://cwe.mitre.org/data/definitions/253.html).
