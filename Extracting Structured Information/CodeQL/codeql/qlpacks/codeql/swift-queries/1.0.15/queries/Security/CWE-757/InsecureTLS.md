# Insecure TLS configuration
TLS v1.0 and v1.1 versions are known to be vulnerable.


## Recommendation
Use `tls_protocol_version_t.TLSv12` or `tls_protocol_version_t.TLSv13` when configuring `URLSession`.


## Example
Specify a newer `tls_protocol_version_t` explicitly, or omit it completely as the OS will use secure defaults.


```swift
// Set TLS version explicitly
func createURLSession() -> URLSession {
  let config = URLSessionConfiguration.default
  config.tlsMinimumSupportedProtocolVersion = tls_protocol_version_t.TLSv13
  return URLSession(configuration: config)
}

// Use the secure OS defaults
func createURLSession() -> URLSession {
  let config = URLSessionConfiguration.default
  return URLSession(configuration: config)
}

```

## References
* [Apple Platform Security - TLS security](https://support.apple.com/en-gb/guide/security/sec100a75d12/web) [Preventing Insecure Network Connections](https://developer.apple.com/documentation/security/preventing_insecure_network_connections)
* Common Weakness Enumeration: [CWE-757](https://cwe.mitre.org/data/definitions/757.html).
