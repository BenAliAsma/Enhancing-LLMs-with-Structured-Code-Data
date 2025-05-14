# Unsafe deserializer
Deserializing an object from untrusted input may result in security problems, such as denial of service or remote code execution.


## Recommendation
Avoid using an unsafe deserialization framework.


## Example
In this example, a string is deserialized using a `JavaScriptSerializer` with a simple type resolver. Using a type resolver means that arbitrary code may be executed.


```csharp
using System.Web.Script.Serialization;

class Bad
{
    public static object Deserialize(string s)
    {
        JavaScriptSerializer sr = new JavaScriptSerializer(new SimpleTypeResolver());
        // BAD
        return sr.DeserializeObject(s);
    }
}

```
To fix this specific vulnerability, we avoid using a type resolver. In other cases, it may be necessary to use a different deserialization framework.


```csharp
using System.Web.Script.Serialization;

class Good
{
    public static object Deserialize(string s)
    {
        // GOOD
        JavaScriptSerializer sr = new JavaScriptSerializer();
        return sr.DeserializeObject(s);
    }
}

```

## References
* Mu&ntilde;oz, Alvaro and Mirosh, Oleksandr: [JSON Attacks](https://www.blackhat.com/docs/us-17/thursday/us-17-Munoz-Friday-The-13th-Json-Attacks.pdf).
* Common Weakness Enumeration: [CWE-502](https://cwe.mitre.org/data/definitions/502.html).
