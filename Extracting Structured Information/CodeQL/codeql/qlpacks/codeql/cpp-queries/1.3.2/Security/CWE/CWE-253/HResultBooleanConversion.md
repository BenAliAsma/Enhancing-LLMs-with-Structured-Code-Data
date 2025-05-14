# Cast between HRESULT and a Boolean type
This query indicates that an `HRESULT` is being cast to a Boolean type or vice versa.

The typical success value (`S_OK`) of an `HRESULT` equals 0. However, 0 indicates failure for a Boolean type.

Casting an `HRESULT` to a Boolean type and then using it in a test expression will yield an incorrect result.


## Recommendation
To check if a call that returns an `HRESULT` succeeded use the `FAILED` macro.


## Example
In the following example, `HRESULT` is used in a test expression incorrectly as it may yield an incorrect result.


```cpp
LPMALLOC pMalloc;
HRESULT hr = CoGetMalloc(1, &pMalloc);

if (!hr)
{
    // code ...
}

```
To fix this issue, use the `FAILED` macro in the test expression.


## References
* Common Weakness Enumeration: [CWE-253](https://cwe.mitre.org/data/definitions/253.html).
