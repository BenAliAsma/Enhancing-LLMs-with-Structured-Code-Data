# StringBuilder creation in loop
Creating a `StringBuilder` in the body of a loop is inefficient. It is more efficient to create the `StringBuilder` before the loop and reuse the same instance for each iteration. Use the `Clear` method to reset the `StringBuilder`, which reuses its internal buffer and is more efficient. This is particularly important in performance-critical code.


## Recommendation
Create the `StringBuilder` before the loop, and call the `Clear` method within the loop to clear the internal buffer.


## Example
The following example creates a new `StringBuilder` in the body of the loop.


```csharp
static void Main(string[] args)
{
    foreach (var arg in args)
    {
        var sb = new StringBuilder();  // BAD: Creation in loop
        sb.Append("Hello ").Append(arg);
        Console.WriteLine(sb);
    }
}

```
The code has been rewritten so that the same `StringBuilder` is used for every iteration of the loop.


```csharp
static void Main(string[] args)
{
    var sb = new StringBuilder();  // GOOD: Creation outside loop
    foreach (var arg in args)
    {
        sb.Clear();
        sb.Append("Hello ").Append(arg);
        Console.WriteLine(sb);
    }
}

```

## References
* MSDN: [StringBuilder](http://msdn.microsoft.com/en-us/library/system.text.stringbuilder.aspx).
