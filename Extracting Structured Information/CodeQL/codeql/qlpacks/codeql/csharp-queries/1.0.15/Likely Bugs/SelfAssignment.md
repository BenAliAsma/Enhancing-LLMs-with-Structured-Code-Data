# Self-assignment
Assigning a variable to itself is redundant and often an indication of a mistake in the code.


## Recommendation
Check the assignment carefully for mistakes. If the assignment is truly redundant and not simply incorrect then remove it.


## Example
In this example the programmer clearly intends to assign to `this.i` but made a mistake.


```csharp
class SelfAssignment
{
    private int i;
    public SelfAssignment(int i)
    {
        i = i;
    }
}

```
