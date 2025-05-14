# Wrong Uint
Find access to an array with a Uint16 when the array has a size lower than 256.


## Recommendation
Use a int with a lower bit size instead. For instance in this example use a 8 bit int.


## Example

```cpp
void test()
{
    uint16_t j = 256;
    char testSubject[122];

    testSubject[j] = 12; // You can use a uint8 here
}

```
