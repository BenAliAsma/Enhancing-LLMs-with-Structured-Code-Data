# Poor error handling: empty catch block
In some languages an empty catch block might be acceptable in certain rare situations such as when a checked exception is guaranteed not to be thrown. C\# does not have checked exceptions so any empty catch block is normally a mistake or an instance of bad practice.

Ignoring exceptions that should be dealt with in some way is almost always a very bad idea. If an exception gets ignored it can allow an attacker to introduce unexpected behavior into your program.


## Recommendation
Ensure all exceptions are handled correctly.


## Example
In this pseudo code example the program keeps running with the same privileges if it fails to drop to lower privileges.


```csharp
class EmptyCatchBlock
{
    public static void Main(string[] args)
    {
        // ...
        try
        {
            SecurityManager.dropPrivileges();
        }
        catch (PrivilegeDropFailedException e)
        {

        }
        // ...
    }
}

```

## References
* Common Weakness Enumeration: [CWE-390](https://cwe.mitre.org/data/definitions/390.html).
* Common Weakness Enumeration: [CWE-391](https://cwe.mitre.org/data/definitions/391.html).
