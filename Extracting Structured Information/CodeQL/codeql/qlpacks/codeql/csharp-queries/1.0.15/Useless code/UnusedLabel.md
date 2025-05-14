# Unused label
An unused label serves no purpose and clutters the source code. It could be an indication that the code is incomplete, or that the code contains a bug where a `goto` statement uses the wrong label.


## Recommendation
Ensure that all `goto` statements use the correct label, and remove the label if it is no longer needed.

Another solution is to rewrite the code without `goto` statements, which can often be much clearer.


## Example
The following example has an unused label `Error`, which means that the error message is never displayed. The code can be fixed by either jumping to the correct label, or by rewriting the code without `goto` statements.


```csharp
static void Main(string[] args)
{
    using (var file = File.Open("values.xml", FileMode.Create))
    using (var stream = new StreamWriter(file))
    {
        stream.WriteLine("<values>");
        foreach (var arg in args)
        {
            uint value;
            if (UInt32.TryParse(arg, out value))
                stream.WriteLine("    <value>{0}</value>", value);
            else
                goto Finish;  // Should be goto Error
        }
        goto Finish;
        Error:  // BAD: Unused label
        Console.WriteLine("Error: All arguments must be non-negative integers");
        Finish:
        stream.WriteLine("</values>");
    }
}

```

## References
* MSDN, C\# Reference: [goto](https://msdn.microsoft.com/en-us/library/13940fs2.aspx).
