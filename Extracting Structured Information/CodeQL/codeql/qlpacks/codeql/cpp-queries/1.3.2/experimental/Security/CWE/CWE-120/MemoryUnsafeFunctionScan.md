# Scanf function without a specified length
It is bad practice to use any of the `scanf` functions without including a specified length within the format parameter, as it will be vulnerable to buffer overflows.


## Recommendation
Specify a length within the format string parameter, and make this length one less than the size of the buffer, since the last character should be reserved for the NULL terminator.


## Example
The following example demonstrates safe and unsafe uses of `scanf` type functions.


```cpp
///// Library routines /////

int scanf(const char *format, ...);
int sscanf(const char *str, const char *format, ...);
int fscanf(const char *str, const char *format, ...);

///// EXAMPLES /////

int main(int argc, char **argv)
{

    // BAD, do not use scanf without specifying a length first
    char buf1[10];
    scanf("%s", buf1);

    // GOOD, length is specified. The length should be one less than the size of the destination buffer, since the last character is the NULL terminator.
    char buf2[20];
    char buf3[10];
    sscanf(buf2, "%9s", buf3);

    // BAD, do not use scanf without specifying a length first
    char file[10];
    fscanf(file, "%s", buf2);

    return 0;
}

```

## References
* Common Weakness Enumeration: [CWE-120](https://cwe.mitre.org/data/definitions/120.html).
