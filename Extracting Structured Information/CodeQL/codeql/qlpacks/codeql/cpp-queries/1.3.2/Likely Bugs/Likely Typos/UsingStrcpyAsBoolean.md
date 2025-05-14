# Use of string copy function in a condition
This query identifies calls to string copy functions used in conditions, either directly or as part of an equality operator or logical operator. The most common string copy functions always return their `destination` parameter and do not have a return value reserved to indicate an error. Therefore, such a function call always evaluates to true in a Boolean context.

The string copy functions that the rule takes into consideration are:

* `strcpy`
* `wcscpy`
* `_mbscpy`
* `strncpy`
* `_strncpy_l`
* `wcsncpy`
* `_wcsncpy_l`
* `_mbsncpy`
* `_mbsncpy_l`
NOTE: It is highly recommended to consider using a more secure version of string manipulation functions such as as `strcpy_s`.


## Recommendation
Check to ensure that the flagged expressions are not typos.

If a string comparison is intended, change the function to the appropriate string comparison function.

If a string copy is really intended, very likely a secure version of the string copy function such as `strcpy_s` was intended instead of the insecure version of the string copy function.


## Example

```cpp
if(strcpy(szbuf1, "Manager") == 0) // most likely strcmp was intended instead of strcpy
{
	// ...
}

```

## References
* Microsoft Code Analysis for C/C++ Warnings: [C6324](https://docs.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2012/ccf4h9w8(v=vs.110))
* Microsoft C library reference: [strcpy, wcscpy, _mbscpy](https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/strcpy-wcscpy-mbscpy)
* US-CERT: [strcpy_s() and strcat_s()](https://www.us-cert.gov/bsi/articles/knowledge/coding-practices/strcpy_s-and-strcat_s)
