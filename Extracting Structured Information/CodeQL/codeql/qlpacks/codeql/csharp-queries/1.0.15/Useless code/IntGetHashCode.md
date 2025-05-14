# Useless call to GetHashCode()
The method `GetHashCode()` on an integer simply returns the original value of the integer. This method call is therefore redundant, inefficient, and obscures the logic of the hash function. Several of the built-in types have this behavior, including `int`, `uint`, `short`, `ushort`, `long`, `ulong`, `byte` and `sbyte`.


## Recommendation
Remove the call to `GetHashCode()`, and review the hash function.


## Example
The following hash function has two problems. Firstly, the calls to `GetHashCode()` are redundant, and secondly, the hash function generates too many collisions.


```csharp
public override int GetHashCode()
{
    return row.GetHashCode() ^ col.GetHashCode();
}

```
These problems are resolved by removing the redundant calls to `GetHashCode()`, and by changing the hash function to generate fewer collisions.


```csharp
public override int GetHashCode()
{
    return unchecked(row * 16777619 + col);
}

```

## References
* MSDN, C\# Reference, [Object.GetHashCode Method](https://msdn.microsoft.com/en-us/library/system.object.gethashcode.aspx).
