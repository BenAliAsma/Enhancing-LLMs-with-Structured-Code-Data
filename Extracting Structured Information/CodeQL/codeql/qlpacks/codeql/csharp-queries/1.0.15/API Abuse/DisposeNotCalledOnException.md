# Dispose may not be called if an exception is thrown during execution
If an exception is thrown between the allocation of an `IDisposable` object and a `Dispose()` call on that object, and the `Dispose()` call is not within a `catch` or `finally` block, then the `Dispose()` call may not execute.


## Recommendation
If possible, wrap the allocation of the object in a `using` block to automatically dispose of the object once the `using` block has completed.

If this is not possible, ensure that `Dispose()` is called on the object. It is usually recommended to call `Dispose()` within a `finally` block, to ensure that the object is disposed of even if an exception is thrown.


## Example
In this example, an `SqlConnection` is created, then an SQL query is run using an `SqlCommand`. The objects are created and disposed of, but if an exception is thrown - for example, by the call to `ExecuteReader` - the method will terminate immediately and `Dispose()` will never be called on `cmd` and `conn`. In the case of the `SqlConnection`, this can result in unmanaged resources associated with the connection being retained, and possibly cause resource exhaustion when trying to create additional future connections.


```csharp
using System;
using System.Data.SqlClient;

class Bad
{
    public SqlDataReader GetAllCustomers()
    {
        var conn = new SqlConnection("connection string");
        conn.Open();

        var cmd = new SqlCommand("SELECT * FROM Customers", conn);
        var ret = cmd.ExecuteReader();

        cmd.Dispose();
        conn.Dispose();

        return ret;
    }
}

```
In the revised example, a pair of `using` statements are used to ensure that both the connection and the command are disposed of after the statements have completed.


```csharp
using System;
using System.Data.SqlClient;

class Good
{
    public SqlDataReader GetAllCustomers()
    {
        using (var conn = new SqlConnection("connection string"))
        {
            conn.Open();
            using (var cmd = new SqlCommand("SELECT * FROM Customers", conn))
            {
                return cmd.ExecuteReader();
            }
        }
    }
}

```

## References
* MSDN: [IDisposable Interface](https://msdn.microsoft.com/en-us/library/system.idisposable.aspx).
* Microsoft: [using Statement (C\# Reference)](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/using-statement).
* Common Weakness Enumeration: [CWE-404](https://cwe.mitre.org/data/definitions/404.html).
* Common Weakness Enumeration: [CWE-459](https://cwe.mitre.org/data/definitions/459.html).
* Common Weakness Enumeration: [CWE-460](https://cwe.mitre.org/data/definitions/460.html).
