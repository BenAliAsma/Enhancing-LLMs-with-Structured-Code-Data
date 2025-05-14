# Comparison is constant
Comparisons which always yield the same result are unnecessary and may indicate a bug in the logic. This can happen when the data type of one of the operands has a limited range of values. For example unsigned integers are always greater than or equal to zero, and `byte` values are always less than 256.

The following expressions always have the same result:

* `Unsigned < 0` is always false,
* `0 > Unsigned` is always false,
* `0 &le; Unsigned` is always true,
* `Unsigned &ge; 0` is always true,
* `Unsigned == -1` is always false,
* `Byte < 512` is always true.

## Recommendation
Examine the logic of the program to determine whether the comparison is necessary. Either change the data types, or remove the unnecessary code.


## Example
The following example attempts to count down from `numberOfOrders` to `0`, however the loop never terminates because `order` is an unsigned integer and so the condition `order >= 0` is always true.


```csharp
    for (uint order = numberOfOrders; order >= 0; order--)
      ProcessOrder(order);

```
The solution is to change the type of the variable `order`.


## References
* MSDN Library: [C\# Operators](https://msdn.microsoft.com/en-us/library/6a71f45d.aspx).
