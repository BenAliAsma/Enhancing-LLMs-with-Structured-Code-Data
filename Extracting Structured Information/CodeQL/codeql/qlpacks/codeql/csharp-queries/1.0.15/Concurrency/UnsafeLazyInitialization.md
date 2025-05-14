# Double-checked lock is not thread-safe
Double-checked locking requires that the underlying field is `volatile`, otherwise the program can behave incorrectly when running in multiple threads, for example by computing the field twice.


## Recommendation
There are several ways to make the code thread-safe:

1. Avoid double-checked locking, and simply perform everything within the lock statement.
1. Make the field volatile using the `volatile` keyword.
1. Use the `System.Lazy` class, which is guaranteed to be thread-safe. This can often lead to more elegant code.
1. Use `System.Threading.LazyInitializer`.

## Example
The following code defines a property called `Name`, which calls the method `LoadNameFromDatabase` the first time that the property is read, and then caches the result. This code is efficient but will not work properly if the property is accessed from multiple threads, because `LoadNameFromDatabase` could be called several times.


```csharp
string name;

public string Name
{
    get
    {
        // BAD: Not thread-safe
        if (name == null)
            name = LoadNameFromDatabase();
        return name;
    }
}

```
A common solution to this is *double-checked locking*, which checks whether the stored value is `null` before locking the mutex. This is efficient because it avoids a potentially expensive lock operation if a value has already been assigned to `name`.


```csharp
string name;    // BAD: Not thread-safe

public string Name
{
    get
    {
        if (name == null)
        {
            lock (mutex)
            {
                if (name == null)
                    name = LoadNameFromDatabase();
            }
        }
        return name;
    }
}

```
However this code is incorrect because the field `name` isn't volatile, which could result in `name` being computed twice on some systems.

The first solution is to simply avoid double-checked locking (Recommendation 1):


```csharp
string name;

public string Name
{
    get
    {
        lock (mutex)    // GOOD: Thread-safe
        {
            if (name == null)
                name = LoadNameFromDatabase();
            return name;
        }
    }
}

```
Another fix would be to make the field volatile (Recommendation 2):


```csharp
volatile string name;    // GOOD: Thread-safe

public string Name
{
    get
    {
        if (name == null)
        {
            lock (mutex)
            {
                if (name == null)
                    name = LoadNameFromDatabase();
            }
        }
        return name;
    }
}

```
It may often be more elegant to use the class `System.Lazy`, which is automatically thread-safe (Recommendation 3):


```csharp
Lazy<string> name;    // GOOD: Thread-safe

public Person()
{
    name = new Lazy<string>(LoadNameFromDatabase);
}

public string Name => name.Value;

```

## References
* MSDN: [Lazy&lt;T&gt; Class](https://docs.microsoft.com/en-us/dotnet/api/system.lazy-1).
* MSDN: [LazyInitializer.EnsureInitialized Method](https://docs.microsoft.com/en-us/dotnet/api/system.threading.lazyinitializer.ensureinitialized).
* MSDN: [Implementing Singleton in C\#](https://msdn.microsoft.com/en-us/library/ff650316.aspx).
* MSDN Magazine: [The C\# Memory Model in Theory and Practice](https://msdn.microsoft.com/magazine/jj863136).
* MSDN, C\# Reference: [volatile](https://msdn.microsoft.com/en-us/library/x13ttww7.aspx).
* Wikipedia: [Double-checked locking](https://en.wikipedia.org/wiki/Double-checked_locking).
* Common Weakness Enumeration: [CWE-609](https://cwe.mitre.org/data/definitions/609.html).
