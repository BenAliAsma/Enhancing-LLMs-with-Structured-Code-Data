# Server-side request forgery
Directly incorporating user input into a HTTP request without validating the input can facilitate Server Side Request Forgery (SSRF) attacks. In these attacks, the server may be tricked into making a request and interacting with an attacker-controlled server.


## Recommendation
To guard against SSRF attacks, it is advisable to avoid putting user input directly into the request URL. Instead, maintain a list of authorized URLs on the server; then choose from that list based on the user input provided.


## Example
The following example shows an HTTP request parameter being used directly in a forming a new request without validating the input, which facilitates SSRF attacks. It also shows how to remedy the problem by validating the user input against a known fixed string.


```csharp
namespace RequestForgery.Controllers
{
    public class SSRFController : Controller
    {
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<ActionResult> Bad(string url)
        {
            var request = new HttpRequestMessage(HttpMethod.Get, url);

            var client = new HttpClient();
            await client.SendAsync(request);

            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<ActionResult> Good(string url)
        {
            string baseUrl = "www.mysecuresite.com/";
            if (url.StartsWith(baseUrl))
            {
                var request = new HttpRequestMessage(HttpMethod.Get, url);
                var client = new HttpClient();
                await client.SendAsync(request);

            }

            return View();
        }
    }
}
```

## References
* [OWASP SSRF](https://owasp.org/www-community/attacks/Server_Side_Request_Forgery)
* Common Weakness Enumeration: [CWE-918](https://cwe.mitre.org/data/definitions/918.html).
