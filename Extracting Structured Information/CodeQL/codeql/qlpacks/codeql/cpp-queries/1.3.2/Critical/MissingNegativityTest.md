# Unchecked return value used as offset
This query finds pointer arithmetic expressions that use a value returned from a function without checking that the value is positive. Most pointer arithmetic and almost all array element accesses use a positive value for offsets. A negative value is likely to be a defect in the returning function. Checking pointer offsets (particularly if they derive from user input) is necessary to avoid buffer overruns.

The query looks only at the return values of functions that may return a negative value (not all functions).

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the actual branch taken in conditional statements such as "if" without running the program with all possible input data. This means that it is not possible to determine if a particular statement is going to be executed.

## Recommendation
Review the function. Determine whether it needs to check that the value is positive before performing pointer arithmetic.


## Example
The example below shows an example of this problem. There is no check to ensure that the value of `recordIdx` is positive and safe to use as an array offset.


```cpp
Record records[SIZE] = ...;

int f() {
    int recordIdx = 0;
    recordIdx = readUserInput(); //recordIdx is returned from a function
        // there is no check so it could be negative
    doFoo(&(records[recordIdx])); //but is not checked before use as an array offset
}


```

## References
* cplusplus.com: [Pointers](http://www.cplusplus.com/doc/tutorial/pointers/).
* SEI CERT C Coding Standard: [ARR30-C. Do not form or use out-of-bounds pointers or array subscripts](https://wiki.sei.cmu.edu/confluence/display/c/ARR30-C.+Do+not+form+or+use+out-of-bounds+pointers+or+array+subscripts).
* Common Weakness Enumeration: [CWE-823](https://cwe.mitre.org/data/definitions/823.html).
