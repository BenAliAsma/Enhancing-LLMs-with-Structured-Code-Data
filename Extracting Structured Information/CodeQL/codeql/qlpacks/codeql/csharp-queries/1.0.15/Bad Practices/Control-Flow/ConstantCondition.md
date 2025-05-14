# Constant condition
A condition that always evaluates to `true` or always evaluates to `false` can be removed, thereby simplifying the program logic. If the condition is a loop condition, consider rewriting the loop using bounded iteration (for example, a `foreach` loop), if possible.


## Recommendation
Avoid constant conditions where possible, and either eliminate the conditions or replace them.


## Example
In the following example, the condition `a > a` is constantly false, so `Max(x, y)` always returns `x`.


```csharp
class Bad
{
    public int Max(int a, int b)
    {
        return a > a ? a : b;
    }
}

```
The revised example replaces the condition with `a > b`.


```csharp
class Good
{
    public int Max(int a, int b)
    {
        return a > b ? a : b;
    }
}

```
