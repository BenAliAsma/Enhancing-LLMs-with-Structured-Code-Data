# Misleading indentation
A control structure (`if` statements and loops) has a body that is either a block of statements or a single statement. The second option may be indicated by omitting the braces: `{` and `}`.

However, omitting the braces can lead to confusion, especially if the indentation of the code suggests that multiple statements are within the body of a control structure when in fact they are not.


## Recommendation
It is usually considered good practice to include braces for all control structures in C\#. This is because it makes it easier to maintain the code later. For example, it's easy to see at a glance which part of the code is in the scope of an `if` statement, and adding more statements to the body of the `if` statement is less error-prone.

You should also ensure that the indentation of the code is consistent with the actual flow of control, so that it does not confuse programmers.


## Example
In the example below, the `if` statement checks whether the item `i` is `null` before adding it to the list. However the `if` statement does not guard the call to `Console.Out.WriteLine`, resulting in a `NullReferenceException` whenever `null` is passed to the function `AddItem`.


```csharp
void AddItem(Object i)
{
    if (i != null)
        items.Add(i);
    Console.Out.WriteLine("Item added: " + i.ToString());
}

```
This code is fixed by adding curly braces `{` and `}` around both statements, as shown below.


```csharp
void AddItem(Object i)
{
    if (i != null)
    {
        items.Add(i);
        Console.Out.WriteLine("Item added: " + i.ToString());
    }
}

```

## References
* MSDN Documentation: [if-else (C\# Reference)](http://msdn.microsoft.com/en-us/library/5011f09h.aspx)
