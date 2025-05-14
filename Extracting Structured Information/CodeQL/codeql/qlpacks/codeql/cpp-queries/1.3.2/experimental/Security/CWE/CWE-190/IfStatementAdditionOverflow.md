# Integer addition may overflow inside if statement
Detects `if (a+b>c) a=c-b`, which incorrectly implements `a = min(a,c-b)` if `a+b` overflows.

Also detects variants such as `if (b+a>c) a=c-b` (swapped terms in addition), `if (a+b>c) { a=c-b }` (assignment inside block), `c<a+b` (swapped operands), and `>=`, `<`, `<=` instead of `>` (all operators).

This integer overflow is the root cause of the buffer overflow in the SHA-3 reference implementation (CVE-2022-37454).


## Recommendation
Replace by `if (a>c-b) a=c-b`. This avoids the overflow and makes it easy to see that `a = min(a,c-b)`.


## References
* CVE-2022-37454: [The Keccak XKCP SHA-3 reference implementation before fdc6fef has an integer overflow and resultant buffer overflow that allows attackers to execute arbitrary code or eliminate expected cryptographic properties. This occurs in the sponge function interface.](https://nvd.nist.gov/vuln/detail/CVE-2022-37454)
* GitHub Advisory Database: [CVE-2022-37454: Buffer overflow in sponge queue functions](https://github.com/advisories/GHSA-6w4m-2xhg-2658)
