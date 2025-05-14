# Bad parity check
Avoid using `x % 2 == 1` or `x % 2 > 0` to check whether a number `x` is odd, or `x % 2 != 1` to check whether it is even. Such code does not work for negative numbers. For example, `-5 % 2` equals `-1`, not `1`.


## Recommendation
Consider using `x % 2 != 0` to check for odd and `x % 2 == 0` to check for even.


## Example
-9 is an odd number but this example does not detect it as one. This is because `-9 % 2 ` is -1, not 1.


```csharp
class CheckOdd
{
    private static bool IsOdd(int x)
    {
        return x % 2 == 1;
    }

    public static void Main(String[] args)
    {
        Console.Out.WriteLine(IsOdd(-9)); // prints False
    }
}

```
It would be better to check if the number is even and then invert that check.


```csharp
class CheckOdd
{
    private static bool IsOdd(int x)
    {
        return x % 2 != 0;
    }

    public static void Main(String[] args)
    {
        Console.Out.WriteLine(IsOdd(-9)); // prints True
    }
}

```

## References
* MSDN Library: [% Operator (C\# Reference)](http://msdn.microsoft.com/en-us/library/0w4e0fzs.aspx).
* Wikipedia: [Modulo Operation - Common pitfalls](http://en.wikipedia.org/wiki/Modulo_operation#Common_pitfalls).
