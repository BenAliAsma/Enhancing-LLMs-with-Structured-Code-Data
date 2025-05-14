# Magic strings: use defined constant
This rule finds strings that are used where a named constant already exists instead.


## Recommendation
Consider replacing the string with the existing named constant.


## Example

```csharp
class MagicStringsUseConstant
{
    private const string Target = "127.0.0.1";
    public static void send(string ip, string message)
    {
        // ...
    }

    public static void testmessage()
    {
        send("127.0.0.1", "test message");
        // BAD: the named constant "Target" should have been used.
    }

    public static void Main(string[] args)
    {
        testmessage();
        send("127.0.0.1", "hello world");
        // BAD: the named constant "Target" should have been used.
    }
}

```

## References
* Robert C. Martin, *Clean Code - A handbook of agile software craftsmanship*, p. 295. Prentice Hall, 2008.
