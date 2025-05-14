# Null dereference from a function result
This rule finds a dereference of a function parameter, whose value comes from another function call that may return NULL, without checks in the meantime.


## Recommendation
A check should be added between the return of the function which may return NULL, and its use by the function dereferencing ths pointer.


## Example

```cpp
char * create (int arg) {
    if (arg > 42) {
        // this function may return NULL
        return NULL;
    }
    char * r = malloc(arg);
    snprintf(r, arg -1, "Hello");
    return r;
}

void process(char *str) {
    // str is dereferenced
    if (str[0] == 'H') {
        printf("Hello H\n");
    }
}

void test(int arg) {
    // first function returns a pointer that may be NULL
    char *str = create(arg);
    // str is not checked for nullness before being passed to process function
    process(str);
}

```

## References
* [ Null Dereference ](https://www.owasp.org/index.php/Null_Dereference)
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
