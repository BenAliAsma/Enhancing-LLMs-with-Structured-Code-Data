# Unsynchronized access to static collection member in non-static context
For performance reasons, most collection classes in the standard library are not thread-safe, instead requiring the user to guarantee they are used from at most one thread at a time by external locking or data structure invariants.

For example, the behavior of `Dictionary` when a write happens concurrently with another write or a read is undefined, and frequently leads to data corruption and can lead to issues as serious as livelock.


## Recommendation
If a static data member such as a `Dictionary` is likely to be accessed from multiple threads, ensure that either it is of a concurrency-safe collection type, or that all reads and writes are guarded by a suitable lock or monitor.


## Example
The following code uses a static dictionary to store properties, but provides unsynchronized access to that dictionary. This means that multiple threads can access the dictionary, potentially leading to a race condition.


```csharp
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading;

public class Configuration
{
    public static Dictionary<string, string> properties = new Dictionary<string, string>();

    // called concurrently elsewhere
    public string getProperty(string key)
    {
        // BAD: unsynchronized access to static collection
        return dict["foo"];
    }
}

```

## References
* MSDN, C\# Reference: [Dictionary: Thread safety](https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.dictionary-2?#thread-safety).
* Common Weakness Enumeration: [CWE-362](https://cwe.mitre.org/data/definitions/362.html).
* Common Weakness Enumeration: [CWE-567](https://cwe.mitre.org/data/definitions/567.html).
