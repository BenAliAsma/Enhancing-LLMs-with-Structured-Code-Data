# Call to GC.Collect()
Explicitly forcing garbage collection is not efficient and is almost never necessary outside of benchmarking scenarios.


## Recommendation
Remove the explicit call to `GC.Collect()` and run a memory profiler to optimize your application's memory usage. If your application uses unmanaged resources and calls ` GC.Collect()` to force finalizers to run, it is better to implement the `IDisposable ` pattern and use `try`/`finally` clauses to make sure that unmanaged resources are disposed of even if an exception interrupts your application.


## Example

```csharp
using System;

class Bad
{
    void M()
    {
        GC.Collect();
    }
}

```

## References
* MSDN: [The IDisposable interface](http://msdn.microsoft.com/en-us/library/system.idisposable.aspx).
* Microsoft: [Profile Memory Usage in Visual Studio](https://docs.microsoft.com/en-us/visualstudio/profiling/memory-usage).
