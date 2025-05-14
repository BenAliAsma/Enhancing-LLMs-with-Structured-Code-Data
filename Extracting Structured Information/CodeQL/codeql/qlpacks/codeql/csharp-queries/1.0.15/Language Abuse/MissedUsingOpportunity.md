# Missed 'using' opportunity
It is good practice (and often essential for correctness) to dispose of resources (for example file handles, graphics handles or database connections) when a program no longer needs them. For resources that are used only within a single method, a common idiom is to enclose the code that uses the resource in a `try` block and dispose of the resource in the `try`'s ` finally ` block. This idiom is in fact so common that C\# provides a shorter, tidier syntax for it in the form of the `using` statement.


## Recommendation
Given the explicit language support provided in this case, it is more idiomatic to use the ` using` statement in preference to the `try`-`finally` technique; it also helps to clearly communicate the intent of your code to other programmers.


## Example
In this example a `try` block is used to ensure resources are disposed of even if the program throws an exception.


```csharp
class MissedUsingOpportunity
{
    static void Main(string[] args)
    {
        StreamReader reader = null;
        try
        {
            reader = File.OpenText("input.txt");
            // ...
        }
        finally
        {
            if (reader != null)
            {
                ((IDisposable)reader).Dispose();
            }
        }
    }
}

```
The example can be significantly simplified by making use of the `using` block instead.


```csharp
class MissedUsingOpportunityFix
{
    static void Main(string[] args)
    {
        using (StreamReader reader = File.OpenText("input.txt"))
        {
            // ...
        }
    }
}

```

## References
* MSDN: [using Statement](http://msdn.microsoft.com/en-us/library/yh598w02(v=vs.80).aspx).
* J. Albahari and B. Albahari, *C\# 4.0 in a Nutshell - The Definitive Reference*, p. 138.
