# Unused method
Dead methods are often deprecated pieces of code, and should be removed as they may increase object code size, decrease code maintainability, and create the possibility of misuse. A method is considered dead if it is private and not directly called, or accessed, or only ever called from other dead methods.


## Recommendation
Consider removing the method or cluster of methods.


## Example
Both private methods in this example are dead. `A` is dead because it is never called and `B` is dead because it is only called from the dead method `A`.


```csharp
class UnusedMethod
{
    private void A()
    {
        B();
    }

    private void B()
    {

    }
}

```
