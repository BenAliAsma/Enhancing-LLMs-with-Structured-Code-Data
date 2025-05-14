# Guarded Free
The `free` function, which deallocates heap memory, may accept a NULL pointer and take no action. Therefore, it is unnecessary to check the argument for the value of NULL before a function call to `free`. As such, these guards may hinder performance and readability.


## Recommendation
A function call to `free` should not depend upon the value of its argument. Delete the condition preceding a function call to `free` when its only purpose is to check the value of the pointer to be freed.


## Example

```cpp
void test()
{
    char *foo = malloc(100);

    // BAD
    if (foo)          
        free(foo);

    // GOOD
    free(foo);
}
```
In this example, the condition checking the value of `foo` can be deleted.


## References
* The Open Group Base Specifications Issue 7, 2018 Edition: [free - free allocated memory](https://pubs.opengroup.org/onlinepubs/9699919799/functions/free.html).
