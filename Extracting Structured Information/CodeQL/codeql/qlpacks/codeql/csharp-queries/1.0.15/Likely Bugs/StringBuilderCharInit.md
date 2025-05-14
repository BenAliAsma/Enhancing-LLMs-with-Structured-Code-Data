# Character passed to StringBuilder constructor
Passing a character to the constructor of `StringBuilder` is probably intended to insert the character into the string. In fact, however, the character value is converted to an integer and interpreted as the internal buffer's initial capacity, so the character value is not inserted into the string.


## Example
The following example shows a `ToString()` method which formats the contents of an array. However, the expression `new StringBuilder('(')` does not add the character `'('` to the string `str` but merely initializes the size of the buffer, so the resulting string does not contain the leading `'('` character.


```csharp
public override string ToString()
{
    var str = new StringBuilder('('); // BAD: Character value.
    for (int i = 0; i < values.Length; ++i)
    {
        if (i > 0) str.Append(',');
        str.Append(values[i]);
    }
    str.Append(')');
    return str.ToString();
}

```
Note that passing a character to `Append()`, on the other hand, is unproblematic.

The problem can be fixed by initializing the `StringBuilder` with a string, which *does* put `"("` at the start of the string.


```csharp
public override string ToString()
{
    var str = new StringBuilder("("); // GOOD: String value.
    for (int i = 0; i < values.Length; ++i)
    {
        if (i > 0) str.Append(',');
        str.Append(values[i]);
    }
    str.Append(')');
    return str.ToString();
}

```

## Recommendation
If the character used to initialize the buffer is a character literal, simply replace it with the corresponding string literal. So, in our example, replace `new StringBuilder('(')` with `new StringBuilder("(")`. If the character is not a literal value, use `ToString()` to convert it to a string, or use an additional call to `Append()` to insert the value into the string.


## References
* MSDN: [StringBuilder Class](https://msdn.microsoft.com/en-us/library/system.text.stringbuilder.aspx)
