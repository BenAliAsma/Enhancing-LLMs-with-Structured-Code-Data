# Untrusted network-to-host usage
Data received over a network connection may be received in a different byte order than the byte order used by the local host, making the data difficult to process. To address this, data received over the wire is usually converted to host byte order by a call to a network-to-host byte order function, such as `ntohl`.

The use of a network-to-host byte order function is therefore a good indicator that the returned value is unvalidated data retrieved from the network, and should not be used without further validation. In particular, the returned value should not be used as an array index or array length value without validation, as this could result in a buffer overflow vulnerability.


## Recommendation
Validate data returned by network-to-host byte order functions before use and especially before using the value as an array index or bound.


## Example
In the example below, network data is retrieved and passed to `ntohl` to convert it to host byte order. The data is then used as an index in an array access expression. However, there is no validation that the data returned by `ntohl` is within the bounds of the array, which could lead to reading outside the bounds of the buffer.


```cpp
int get_number_from_network();

int process_network(int[] buff, int buffSize) {
  int i = ntohl(get_number_from_network());
  return buff[i];
}
```
In the corrected example, the returned data is validated against the known size of the buffer, before being used as an array index.


```cpp
uint32_t get_number_from_network();

int process_network(int[] buff, uint32_t buffSize) {
  uint32_t i = ntohl(get_number_from_network());
  if (i < buffSize) {
    return buff[i];
  } else {
    return -1;
  }
}
```

## References
* [ ntohl - winsock reference ](https://docs.microsoft.com/en-us/windows/desktop/api/winsock/nf-winsock-ntohl)
* [ ntohl - Linux man page ](https://linux.die.net/man/3/ntohl)
