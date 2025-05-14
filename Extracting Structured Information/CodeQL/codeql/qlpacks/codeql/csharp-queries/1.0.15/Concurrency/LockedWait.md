# A lock is held during a wait
Holding a lock during a call to `System.Threading.Monitor.Wait()` can lead to performance problems or deadlock because the lock can prevent other threads from running. This can be caused by nesting the call to `System.Threading.Monitor.Wait()` in two `lock` statements, or by waiting on a different object to the one which is locked.

Synchronized methods (with the attribute `[MethodImpl(MethodImplOptions.Synchronized)]`) can also cause problems, because they are equivalent to `lock(this)` or `lock(typeof(Class))`.


## Recommendation
Ensure that the call to `System.Threading.Monitor.Wait()` occurs within a single `lock` statement and ensure that the argument to `System.Threading.Monitor.Wait()` matches the variable in the `lock` statement.


## Example
The following example locks two variables, `countLock` and `textLock`, then calls `System.Threading.Monitor.Wait()`. However for the duration of `Wait()`, the variable `countLock` is locked, which deadlocks the program.


```csharp
class Message
{
    readonly Object countLock = new Object();
    readonly Object textLock = new Object();

    int count;
    string text;

    public void Print()
    {
        lock (countLock)
        {
            lock (textLock)
            {
                while (text == null)
                    System.Threading.Monitor.Wait(textLock);
                System.Console.Out.WriteLine(text + "=" + count);
            }
        }
    }

    public string Text
    {
        set
        {
            lock (countLock)
            {
                lock (textLock)
                {
                    ++count;
                    text = value;
                    System.Threading.Monitor.Pulse(textLock);
                }
            }
        }
    }
}

```
The problem can be fixed by moving the `lock` statement so that `countLock` is not locked for the duration of the wait.


```csharp
class Message
{
    readonly Object countLock = new Object();
    readonly Object textLock = new Object();

    int count;
    string text;

    public void Print()
    {
        lock (textLock)
        {
            while (text == null)
                System.Threading.Monitor.Wait(textLock);
            lock (countLock)
                System.Console.Out.WriteLine(text + "=" + count);
        }
    }

    public string Text
    {
        set
        {
            lock (countLock)
            {
                lock (textLock)
                {
                    ++count;
                    text = value;
                    System.Threading.Monitor.Pulse(textLock);
                }
            }
        }
    }
}

```

## References
* MSDN, C\# Reference: [lock Statement](http://msdn.microsoft.com/en-gb/library/c5kehkcz.aspx).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
* Common Weakness Enumeration: [CWE-833](https://cwe.mitre.org/data/definitions/833.html).
