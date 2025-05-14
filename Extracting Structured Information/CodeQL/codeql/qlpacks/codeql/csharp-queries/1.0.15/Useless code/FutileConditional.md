# Futile conditional
This rule finds If-statements where the "then" branch is empty and there is no "else" branch. These statements are usually unimplemented skeleton code that should be implemented, or real unused code that should be removed.


## Recommendation
There might be missing statements in the then-branch or the If-statement might be able to be removed completely.


## Example

```csharp
class FutileConditional
{
    static void Main(string[] args)
    {
        if (args.Length > 10) ; // BAD
        if (args.Length > 8)
        {
            // BAD
        }
        if (args.Length > 6)
        {
            // GOOD: because of else-branch
        }
        else
        {
            System.Console.WriteLine("hello");
        }
    }
}

```

## References
* MSDN, C\# Reference, [if-else](http://msdn.microsoft.com/en-us/library/5011f09h(v=vs.110).aspx)
