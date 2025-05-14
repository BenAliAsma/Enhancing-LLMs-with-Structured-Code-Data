# XML external entity expansion
Parsing untrusted XML files with a weakly configured XML parser may lead to an XML external entity (XXE) attack. This type of attack uses external entity references to access arbitrary files on a system, carry out denial-of-service (DoS) attacks, or server-side request forgery. Even when the result of parsing is not returned to the user, DoS attacks are still possible and out-of-band data retrieval techniques may allow attackers to steal sensitive data.


## Recommendation
The easiest way to prevent XXE attacks is to disable external entity handling when parsing untrusted data. How this is done depends on the library being used. Note that some libraries, such as recent versions of `libxml`, disable entity expansion by default, so unless you have explicitly enabled entity expansion, no further action needs to be taken.


## Example
The following example uses the `Xerces-C++` XML parser to parse a string `data`. If that string is from an untrusted source, this code may be vulnerable to an XXE attack, since the parser is constructed in its default state with `setDisableDefaultEntityResolution` set to `false`:


```cpp

XercesDOMParser *parser = new XercesDOMParser();

parser->parse(data); // BAD (parser is not correctly configured, may expand external entity references)

```
To guard against XXE attacks, the `setDisableDefaultEntityResolution` option should be set to `true`.


```cpp

XercesDOMParser *parser = new XercesDOMParser();

parser->setDisableDefaultEntityResolution(true);
parser->parse(data);

```

## References
* OWASP: [XML External Entity (XXE) Processing](https://www.owasp.org/index.php/XML_External_Entity_(XXE)_Processing).
* OWASP: [XML External Entity Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/XML_External_Entity_Prevention_Cheat_Sheet.html).
* Timothy Morgen: [XML Schema, DTD, and Entity Attacks](https://research.nccgroup.com/2014/05/19/xml-schema-dtd-and-entity-attacks-a-compendium-of-known-techniques/).
* Timur Yunusov, Alexey Osipov: [XML Out-Of-Band Data Retrieval](https://www.slideshare.net/qqlan/bh-ready-v4).
* Common Weakness Enumeration: [CWE-611](https://cwe.mitre.org/data/definitions/611.html).
