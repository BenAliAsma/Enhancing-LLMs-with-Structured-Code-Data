# Inclusion of functionality from an untrusted source
Including a resource from an untrusted source or using an untrusted channel may allow an attacker to include arbitrary code in the response. When including an external resource (for example, a `script` element or an `iframe` element) on a page, it is important to ensure that the received data is not malicious.

When including external resources, it is possible to verify that the responding server is the intended one by using an `https` URL. This prevents a MITM (man-in-the-middle) attack where an attacker might have been able to spoof a server response.

Even when `https` is used, an attacker might still compromise the server. When you use a `script` element, you can check for subresource integrity - that is, you can check the contents of the data received by supplying a cryptographic digest of the expected sources to the `script` element. The script will only load sources that match the digest and an attacker will be unable to modify the script even when the server is compromised.

Subresource integrity (SRI) checking is commonly recommended when importing a fixed version of a library - for example, from a CDN (content-delivery network). Then, the fixed digest of that version of the library can easily be added to the `script` element's `integrity` attribute.

A dynamic service cannot be easily used with SRI. Nevertheless, it is possible to list multiple acceptable SHA hashes in the `integrity` attribute, such as those for the content generated for major browers used by your users.

See the \[\`CUSTOMIZING.md\`\](https://github.com/github/codeql/blob/main/javascript/ql/src/Security/CWE-830/CUSTOMIZING.md) file in the source code for this query for information on how to extend the list of hostnames required to use SRI by this query.


## Recommendation
When an `iframe` element is used to embed a page, it is important to use an `https` URL.

When using a `script` element to load a script, it is important to use an `https` URL and to consider checking subresource integrity.


## Example
The following example loads the jQuery library from the jQuery CDN without using `https` and without checking subresource integrity.


```html
<html>
    <head>
        <title>jQuery demo</title>
        <script src="http://code.jquery.com/jquery-3.6.0.slim.min.js" crossorigin="anonymous"></script>
    </head>
    <body>
        ...
    </body>
</html>
```
Instead, loading jQuery from the same domain using `https` and checking subresource integrity is recommended, as in the next example.


```html
<html>
    <head>
        <title>jQuery demo</title>
        <script src="https://code.jquery.com/jquery-3.6.0.slim.min.js" integrity="sha256-u7e5khyithlIdTpu22PHhENmPcRdFiHRjhAuHcs05RI=" crossorigin="anonymous"></script>
    </head>
    <body>
        ...
    </body>
</html>
```

## References
* MDN: [Subresource Integrity](https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity)
* Smashing Magazine: [Understanding Subresource Integrity](https://www.smashingmagazine.com/2019/04/understanding-subresource-integrity/)
* Common Weakness Enumeration: [CWE-830](https://cwe.mitre.org/data/definitions/830.html).
