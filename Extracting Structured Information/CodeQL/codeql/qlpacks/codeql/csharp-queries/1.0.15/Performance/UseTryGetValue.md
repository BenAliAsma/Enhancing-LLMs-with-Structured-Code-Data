# Inefficient use of ContainsKey
Using the `ContainsKey` method to check whether a dictionary contains a value before getting the value is inefficient because it performs two operations on the dictionary. It is simpler and more efficient to combine the operations using the `TryGetValue` method.


## Recommendation
Replace the two operations with a single call to `TryGetValue`.


## Example
This code first checks whether `ip` is in the `hostnames` table, before looking up the value.


```csharp
// BAD: Two operations on the hostnames table.

if(hostnames.ContainsKey(ip))
  return hostnames[ip];

```
This code performs the same function as the example above, but uses `TryGetValue`, which makes it is more efficient.


```csharp
// GOOD: One operation on the hostnames table.

if(hostnames.TryGetValue(ip, out hostname))
  return hostname;

```

## References
* MSDN: [ContainsKey Method](https://msdn.microsoft.com/en-us/library/kw5aaea4(v=vs.110).aspx), [TryGetValue Method](https://msdn.microsoft.com/en-us/library/bb347013(v=vs.110).aspx).
