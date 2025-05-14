# Useless ?? expression
It is rare, but not impossible, to write a null-coalescing expression whose left and right operands both do the same thing. The consequences are typically unfortunate - since a null-coalescing expression is generally used to do something special when a variable is null, failing to handle that (evidently expected) case often leads to a `NullReferenceException `, either immediately or later in the program's execution.


## Recommendation
Rewrite the `else` part of the null-coalescing expression to provide a suitable substitute if the variable is null.


## Example
This example demonstrates a simple class designed to generate random numbers. At its core is an instance of `Random` called `generator`. In order to be more efficient, the programmer has chosen not to initialize the generator until it is first needed. Due to both sides of the null-coalescing expression being the same, `generator` is never initialized and a ` NullReferenceException` occurs when attempting to call `Next`.


```csharp
class UselessNullCoalescingExpression
{
    private static Random generator;

    private static int RandomNumber()
    {
        // This should probably have said "generator ?? new Random()".
        generator = generator ?? generator;
        return generator.Next();
    }

    static void Main(string[] args)
    {
        Console.WriteLine(RandomNumber());
        Console.WriteLine(RandomNumber());
        Console.WriteLine(RandomNumber());
    }
}

```
