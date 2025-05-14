# Untrusted domain used in script or other content
Content Delivery Networks (CDNs) are used to deliver content to users quickly and efficiently. However, they can change hands or be operated by untrustworthy owners, risking the security of the sites that use them. Some CDN domains are operated by entities that have used CDNs to deliver malware, which this query identifies.

For example, `polyfill.io` was a popular JavaScript CDN, used to support new web browser standards on older browsers. In February 2024 the domain was sold, and in June 2024 it was publicised that the domain had been used to serve malicious scripts. It was taken down later in that month, leaving a window where sites that used the service could have been compromised. The same operator runs several other CDNs, undermining trust in those too.

Including a resource from an untrusted source or using an untrusted channel may allow an attacker to include arbitrary code in the response. When including an external resource (for example, a `script` element) on a page, it is important to ensure that the received data is not malicious.

Even when `https` is used, an untrustworthy operator might deliver malware.

See the \[\`CUSTOMIZING.md\`\](https://github.com/github/codeql/blob/main/javascript/ql/src/Security/CWE-830/CUSTOMIZING.md) file in the source code for this query for information on how to extend the list of untrusted domains used by this query.


## Recommendation
Carefully research the ownership of a Content Delivery Network (CDN) before using it in your application.

If you find code that originated from an untrusted domain in your application, you should review your logs to check for compromise.

To help mitigate the risk of including a script that could be compromised in the future, consider whether you need to use polyfill or another library at all. Modern browsers do not require a polyfill, and other popular libraries were made redundant by enhancements to HTML 5.

If you do need a polyfill service or library, move to using a CDN that you trust.

When you use a `script` or `link` element, you should check for [subresource integrity (SRI)](https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity), and pin to a hash of a version of the service that you can trust (for example, because you have audited it for security and unwanted features). A dynamic service cannot be easily used with SRI. Nevertheless, it is possible to list multiple acceptable SHA hashes in the `integrity` attribute, such as hashes for the content required for the major browsers used by your users.

You can also choose to self-host an uncompromised version of the service or library.


## Example
The following example loads the Polyfill.io library from the `polyfill.io` CDN. This use was open to malicious scripts being served by the CDN.


```html
<html>
    <head>
        <title>Polyfill.io demo</title>
        <script src="https://cdn.polyfill.io/v2/polyfill.min.js" crossorigin="anonymous"></script>
    </head>
    <body>
        ...
    </body>
</html>
```
Instead, load the Polyfill library from a trusted CDN, as in the next example:


```html
<html>
    <head>
        <title>Polyfill demo - Cloudflare hosted with pinned version (but no integrity checking, since it is dynamically generated)</title>
        <script src="https://cdnjs.cloudflare.com/polyfill/v3/polyfill.min.js?version=4.8.0" crossorigin="anonymous"></script>
    </head>
    <body>
        ...
    </body>
</html>
```
If you know which browsers are used by the majority of your users, you can list the hashes of the polyfills for those browsers:


```html
<html>
    <head>
        <title>Polyfill demo - Cloudflare hosted with pinned version (with integrity checking for a *very limited* browser set - just an example!)</title>
        <script src="https://cdnjs.cloudflare.com/polyfill/v3/polyfill.min.js?version=4.8.0" integrity="sha384-i0IGVuZBkKZqwXTD4CH4kcksIbFx7WKFMdxN8zUhLFHpLdELF0ym0jxa6UvLhW8/ sha384-3d4jRKquKl90C9aFG+eH4lPJmtbPHgACWHrp+VomFOxF8lzx2jxqeYkhpRg18UWC" crossorigin="anonymous"></script>
    </head>
    <body>
        ...
    </body>
</html>
```

## References
* Sansec: [Polyfill supply chain attack hits 100K+ sites](https://sansec.io/research/polyfill-supply-chain-attack)
* Cloudflare: [Upgrade the web. Automatically. Delivers only the polyfills required by the user's web browser.](https://cdnjs.cloudflare.com/polyfill)
* Fastly: [New options for Polyfill.io users](https://community.fastly.com/t/new-options-for-polyfill-io-users/2540)
* Wikipedia: [Polyfill (programming)](https://en.wikipedia.org/wiki/Polyfill_(programming))
* MDN Web Docs: [Subresource Integrity](https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity)
* Common Weakness Enumeration: [CWE-830](https://cwe.mitre.org/data/definitions/830.html).
