# Uncontrolled data used in a WebClient
The WebClient class provides a variety of methods for data transmission and communication with a particular URI. Despite of the class' naming convention, the URI scheme can also identify local resources, not only remote ones. Tainted by user-supplied input, the URI can be leveraged to access resources available on the local file system, therefore leading to the disclosure of sensitive information. This can be trivially achieved by supplying path traversal sequences (../) followed by an existing directory or file path.

Sanitization of user-supplied URI values using the `StartsWith("https://")` method is deemed insufficient in preventing arbitrary file reads. This is due to the fact that .NET ignores the protocol handler (https in this case) in URIs like the following: "https://../../../../etc/passwd".


## Recommendation
Validate user input before using it to ensure that is a URI of an external resource and not a local one. Potential solutions:

* Sanitize potentially tainted paths using `System.Uri.IsWellFormedUriString`.

## Example
In the first example, a domain name is read from a `HttpRequest` and then this domain is requested using the method `DownloadString`. However, a malicious user could enter a local path - for example, "../../../etc/passwd" instead of a domain. In the second example, it appears that the user is restricted to the HTTPS protocol handler. However, a malicious user could still enter a local path, since as explained above the protocol handler will be ignored by .net. For example, the string "https://../../../etc/passwd" will result in the code reading the file located at "/etc/passwd", which is the system's password file. This file would then be sent back to the user, giving them access to all the system's passwords.


```csharp
using System;
using System.IO;
using System.Web;
using System.Net;

public class TaintedWebClientHandler : IHttpHandler
{
    public void ProcessRequest(HttpContext ctx)
    {
        String url = ctx.Request.QueryString["domain"];

        // BAD: This could read any file on the filesystem. (../../../../etc/passwd)
		using(WebClient client = new WebClient()) {
			ctx.Response.Write(client.DownloadString(url));
        }

        // BAD: This could still read any file on the filesystem. (https://../../../../etc/passwd)
        if (url.StartsWith("https://")){
            using(WebClient client = new WebClient()) {
                ctx.Response.Write(client.DownloadString(url));
            }
        }

        // GOOD: IsWellFormedUriString ensures that it is a valid URL
        if (Uri.IsWellFormedUriString(url, UriKind.Absolute)){
            using(WebClient client = new WebClient()) {
                ctx.Response.Write(client.DownloadString(url));
            }
        }
    }
}

```

## References
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* Common Weakness Enumeration: [CWE-99](https://cwe.mitre.org/data/definitions/99.html).
* Common Weakness Enumeration: [CWE-23](https://cwe.mitre.org/data/definitions/23.html).
* Common Weakness Enumeration: [CWE-36](https://cwe.mitre.org/data/definitions/36.html).
* Common Weakness Enumeration: [CWE-73](https://cwe.mitre.org/data/definitions/73.html).
