# Deserialization of untrusted data
Deserializing an object from untrusted input may result in security problems, such as denial of service or remote code execution.


## Recommendation
Avoid deserializing objects from an untrusted source, and if not possible, make sure to use a safe deserialization framework.


## Example
In this example, text from an HTML text box is deserialized using a `JavaScriptSerializer` with a simple type resolver. Using a type resolver means that arbitrary code may be executed.


```csharp
using System.Web.UI.WebControls;
using System.Web.Script.Serialization;

class Bad
{
    public static object Deserialize(TextBox textBox)
    {
        JavaScriptSerializer sr = new JavaScriptSerializer(new SimpleTypeResolver());
        // BAD
        return sr.DeserializeObject(textBox.Text);
    }
}

```
To fix this specific vulnerability, we avoid using a type resolver. In other cases, it may be necessary to use a different deserialization framework.


```csharp
using System.Web.UI.WebControls;
using System.Web.Script.Serialization;

class Good
{
    public static object Deserialize(TextBox textBox)
    {
        JavaScriptSerializer sr = new JavaScriptSerializer();
        // GOOD: no unsafe type resolver
        return sr.DeserializeObject(textBox.Text);
    }
}

```
In the following example potentially untrusted stream and type is deserialized using a `DataContractJsonSerializer` which is known to be vulnerable with user supplied types.


```csharp
using System.Runtime.Serialization.Json;
using System.IO;
using System;

class BadDataContractJsonSerializer
{
    public static object Deserialize(string type, Stream s)
    {
        // BAD: stream and type are potentially untrusted
        var ds = new DataContractJsonSerializer(Type.GetType(type));
        return ds.ReadObject(s);
    }
}

```
To fix this specific vulnerability, we are using hardcoded Plain Old CLR Object ([POCO](https://en.wikipedia.org/wiki/Plain_old_CLR_object)) type. In other cases, it may be necessary to use a different deserialization framework.


```csharp
using System.Runtime.Serialization.Json;
using System.IO;
using System;

class Poco
{
    public int Count;

    public string Comment;
}

class GoodDataContractJsonSerializer
{
    public static Poco Deserialize(Stream s)
    {
        // GOOD: while stream is potentially untrusted, the instantiated type is hardcoded
        var ds = new DataContractJsonSerializer(typeof(Poco));
        return (Poco)ds.ReadObject(s);
    }
}

```

## References
* Mu&ntilde;oz, Alvaro and Mirosh, Oleksandr: [JSON Attacks](https://www.blackhat.com/docs/us-17/thursday/us-17-Munoz-Friday-The-13th-Json-Attacks.pdf).
* Common Weakness Enumeration: [CWE-502](https://cwe.mitre.org/data/definitions/502.html).
