# XML is read insecurely
XML documents can contain Document Type Definitions (DTDs), which may define new XML entities. These can be used to perform Denial of Service (DoS) attacks, or resolve to resources outside the intended sphere of control.


## Recommendation
When processing XML documents, ensure that DTD processing is disabled unless absolutely necessary, and if it is necessary, ensure that a secure resolver is used.


## Example
The following example shows an HTTP request parameter being read directly into an `XmlTextReader`. In the current version of the .NET Framework, `XmlTextReader` has DTD processing enabled by default.


```csharp
public class XMLHandler : IHttpHandler
{
    public void ProcessRequest(HttpContext ctx)
    {
        // BAD: XmlTextReader is insecure by default, and the payload is user-provided data
        XmlTextReader reader = new XmlTextReader(ctx.Request.QueryString["document"]);
    ...
  }
}


```
The solution is to set the `DtdProcessing` property to `DtdProcessing.Prohibit`.


## References
* OWASP: [XML External Entity (XXE) Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/XML_External_Entity_Prevention_Cheat_Sheet.html).
* Microsoft Docs: [System.XML: Security considerations](https://msdn.microsoft.com/en-us/library/system.xml.xmlreadersettings(v=vs.110).aspx#Anchor_6).
* Common Weakness Enumeration: [CWE-611](https://cwe.mitre.org/data/definitions/611.html).
* Common Weakness Enumeration: [CWE-827](https://cwe.mitre.org/data/definitions/827.html).
* Common Weakness Enumeration: [CWE-776](https://cwe.mitre.org/data/definitions/776.html).
