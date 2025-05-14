# Resolving XML external entity in user-controlled data
Parsing untrusted XML files with a weakly configured XML parser may lead to an XML External Entity (XXE) attack. This type of attack uses external entity references to access arbitrary files on a system, carry out denial-of-service attacks, or server-side request forgery. Even when the result of parsing is not returned to the user, out-of-band data retrieval techniques may allow attackers to steal sensitive data. Denial of services can also be carried out in this situation.


## Recommendation
The easiest way to prevent XXE attacks is to disable external entity handling when parsing untrusted data. How this is done depends on the library being used. Note that some libraries, such as recent versions of `XMLParser`, disable entity expansion by default, so unless you have explicitly enabled entity expansion, no further action needs to be taken.


## Example
The following example uses the `XMLParser` class to parse a string `data`. If that string is from an untrusted source, this code may be vulnerable to an XXE attack, since the parser is also setting its `shouldResolveExternalEntities` option to `true`:


```swift
let parser = XMLParser(data: remoteData) // BAD (parser explicitly enables external entities)
parser.shouldResolveExternalEntities = true

```
To guard against XXE attacks, the `shouldResolveExternalEntities` option should be left unset or explicitly set to `false`.


```swift
let parser = XMLParser(data: remoteData) // GOOD (parser explicitly disables external entities)
parser.shouldResolveExternalEntities = false

```

## References
* OWASP: [XML External Entity (XXE) Processing](https://www.owasp.org/index.php/XML_External_Entity_(XXE)_Processing).
* OWASP: [XML External Entity Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/XML_External_Entity_Prevention_Cheat_Sheet.html).
* Timothy D. Morgan and Omar Al Ibrahim [XML Schema, DTD, and Entity Attacks: A Compendium of Known Techniques](https://research.nccgroup.com/2014/05/19/xml-schema-dtd-and-entity-attacks-a-compendium-of-known-techniques/).
* Timur Yunusov, Alexey Osipov: [XML Out-Of-Band Data Retrieval](https://www.slideshare.net/qqlan/bh-ready-v4).
* Common Weakness Enumeration: [CWE-611](https://cwe.mitre.org/data/definitions/611.html).
* Common Weakness Enumeration: [CWE-776](https://cwe.mitre.org/data/definitions/776.html).
* Common Weakness Enumeration: [CWE-827](https://cwe.mitre.org/data/definitions/827.html).
