# Redundant null check or missing null check of parameter
This rule finds comparisons of a function parameter to null that occur when in another path the parameter is dereferenced without a guard check. It's likely either the check is not required and can be removed, or it should be added before the dereference so that a null pointer dereference does not occur.


## Recommendation
A check should be added to before the dereference, in a way that prevents a null pointer value from being dereferenced. If it's clear that the pointer cannot be null, consider removing the check instead.


## Example

```cpp
void test(char *arg1, int *arg2) {
    if (arg1[0] == 'A') {
        if (arg2 != NULL) { //maybe redundant
            *arg2 = 42;
        }
    }
    if (arg1[1] == 'B')
    {
        *arg2 = 54; //dereferenced without checking first
    }
}

```

## References
* [ Null Dereference ](https://www.owasp.org/index.php/Null_Dereference)
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
