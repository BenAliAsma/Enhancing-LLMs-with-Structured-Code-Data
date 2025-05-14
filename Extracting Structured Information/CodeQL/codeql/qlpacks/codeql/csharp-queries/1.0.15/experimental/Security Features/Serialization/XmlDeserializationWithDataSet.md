# XML deserialization with a type type derived from DataSet or DataTable
The `DataSet` and `DataTable` types are legacy .NET components that you can use to represent data sets as managed objects.

While `DataSet` and `DataTable` do impose default limitations on the types that are allowed to be present while deserializing XML payloads, `DataSet` and `DataTable` are in general not safe when populated with untrusted input.

Please visit [DataSet and DataTable security guidance](https://go.microsoft.com/fwlink/?linkid=2132227) for more details.


## Recommendation
Please review the [DataSet and DataTable security guidance](https://go.microsoft.com/fwlink/?linkid=2132227) before making use of these types for serialization.


## References
* Microsoft Docs[DataSet and DataTable security guidance](https://go.microsoft.com/fwlink/?linkid=2132227).
