# Nested 'if' statements can be combined
It is unnecessary to nest `if` statements when neither of them has an `else` part. The code can be written more simply by combining the statements into a single `if` statement.


## Recommendation
Combine the `if` statements into a single `if` statement. Combine the conditions using an `&&` operator.

Be sure to check operator precedence and use brackets around each condition where necessary.


## Example
This example shows two `if` statements which are nested.


```csharp
if(connection.Status == Connected)
{
  if(!connection.Authenticated)
  {
    connection.SendAuthRequest();
  }
}

```
Since neither of the statements has an `else` part, the code can be rewritten as follows:


```csharp
if(connection.Status == Connected && !connection.Authenticated)
{
  connection.SendAuthRequest();
}

```

## References
* MSDN: [if-else (C\# Reference)](https://msdn.microsoft.com/en-us/library/5011f09h.aspx).
* MSDN: [C\# Operators](https://msdn.microsoft.com/en-us/library/6a71f45d.aspx).
