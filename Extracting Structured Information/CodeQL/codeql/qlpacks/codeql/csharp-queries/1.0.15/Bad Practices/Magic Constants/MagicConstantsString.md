# Magic strings
This rule finds strings that are used multiple times that should possibly be constants instead. Named constants make it easier to determine the purpose of the string and make the string easy to change if it is used more than once.


## Recommendation
Consider replacing the string with a named constant.


## Example
In the example shown it would be better if the string "127.0.0.1" was made into a named constant.


```csharp
class MagicConstantsString
{
    public static void send(string ip, string message)
    {
        // ...
    }

    public static void testmessage()
    {
        send("127.0.0.1", "test message");
    }

    public static void Main(string[] args)
    {
        testmessage();
        send("127.0.0.1", "hello world");
    }
}

```

## References
* Robert C. Martin, *Clean Code - A handbook of agile software craftsmanship*, p. 295. Prentice Hall, 2008.
