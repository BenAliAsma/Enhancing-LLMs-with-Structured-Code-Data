# String concatenation in loop
This rule finds code that performs string concatenation in a loop using the `+` operator. If the enclosing loop in question is executed many times, the use of the `+` operator may be inefficient.


## Recommendation
It is better to use `System.Text.StringBuilder` for efficiency.


## Example

```csharp
class StringConcatenationInLoop
{
    public static void Main(string[] args)
    {
        String numberList = "";
        for (int i = 0; i <= 100; i++)
        {
            numberList += i + " ";
        }
        Console.WriteLine(numberList);
    }
}

```

## Fix With StringBuilder
This code performs the same function as the example except it uses `StringBuilder` so it is more efficient.


```csharp
class StringConcatenationInLoopFix
{
    public static void Main(string[] args)
    {
        StringBuilder numberList = new StringBuilder();
        for (int i = 0; i <= 100; i++)
        {
            numberList.Append(i);
            numberList.Append(" ");
        }
        Console.WriteLine(numberList);
    }
}

```

## References
* MSDN: [StringBuilder](http://msdn.microsoft.com/en-us/library/system.text.stringbuilder.aspx).
