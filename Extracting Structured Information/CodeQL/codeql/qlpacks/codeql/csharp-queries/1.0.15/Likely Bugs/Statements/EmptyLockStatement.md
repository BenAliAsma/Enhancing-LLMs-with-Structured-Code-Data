# Empty lock statement
Empty lock statements suspend execution until a lock can be acquired, and then release that lock immediately. This can sometimes achieve the desired effect, but there are generally much better ways to accomplish the same thing using event wait handles.

There are also other reasons why you may have an empty lock statement in your code. For example the statement may have originally contained code that has since been removed or code may have been accidentally placed outside the statement when it should be inside it.


## Recommendation
Since there are various causes for an empty lock statement, dealing with the presence of one requires careful analysis and a good understanding of what the code is intended to achieve. In some cases, the problem can be solved by refactoring the code to use more appropriate synchronization mechanisms. In other cases, the lock is redundant and can be removed altogether. In the most common case, the empty lock may just indicate code that is incorrectly synchronized and needs fixing or redesigning.


## Example
In the code below, an empty lock statement is being used to ensure that the assignment to `value` has been completed before it is printed by method `Y`.


```csharp
class EmptyLockStatement
{
    private static readonly object locker = new object();
    private static Thread threadX;
    private static Thread threadY;
    private static int value;

    static void Main(string[] args)
    {
        threadX = new Thread(X);
        threadY = new Thread(Y);
        threadX.Start();

        threadX.Join();
        while (threadY.ThreadState == ThreadState.Unstarted) ;
        threadY.Join();
    }

    static void X()
    {
        lock (locker)
        {
            threadY.Start();
            value = 1;
        }
    }

    static void Y()
    {
        lock (locker) { }
        Console.WriteLine(value);
    }
}

```
The previous example works, but can be implemented in a much better and more idiomatic way using an event wait handle.


```csharp
class EmptyLockStatementFix
{
    private static Thread threadX;
    private static Thread threadY;
    private static int value;
    private static EventWaitHandle waitHandle = new AutoResetEvent(false);

    static void Main(string[] args)
    {
        threadX = new Thread(X);
        threadY = new Thread(Y);
        threadX.Start();

        threadX.Join();
        while (threadY.ThreadState == ThreadState.Unstarted) ;
        threadY.Join();
    }

    static void X()
    {
        threadY.Start();
        value = 1;
        waitHandle.Set();
    }

    static void Y()
    {
        waitHandle.WaitOne();
        Console.WriteLine(value);
    }
}

```
