# Cleartext transmission of sensitive information
Sensitive information that is transmitted without encryption may be accessible to an attacker.


## Recommendation
Ensure that sensitive information is always encrypted before being transmitted over the network. In general, decrypt sensitive information only at the point where it is necessary for it to be used in cleartext. Avoid transmitting sensitive information when it is not necessary to.


## Example
The following example shows three cases of transmitting information. In the 'BAD' case, the data transmitted is sensitive (a credit card number) and is not encrypted. In the 'GOOD' cases, the data is either not sensitive, or is protected with encryption. When encryption is used, take care to select a secure modern encryption algorithm, and put suitable key management practices into place.


```swift
import CryptoKit

private func encrypt(_ text: String, _ encryptionKey: SymmetricKey) -> String {
	let sealedBox = try! AES.GCM.seal(Data(text.utf8), using: encryptionKey)
	return sealedBox.combined!.base64EncodedString()
}

func transmitMyData(connection : NWConnection, faveSong : String, creditCardNo : String, encryptionKey: SymmetricKey) {
	// ...

	// GOOD: not sensitive information
	connection.send(content: faveSong, completion: .idempotent)

	// BAD: sensitive information saved in cleartext
	connection.send(content: creditCardNo, completion: .idempotent)

	// GOOD: encrypted sensitive information saved
	connection.send(content: encrypt(creditCardNo, encryptionKey), completion: .idempotent)

	// ...
}

```

## References
* OWASP Top 10:2021: [A02:2021 � Cryptographic Failures](https://owasp.org/Top10/A02_2021-Cryptographic_Failures/).
* OWASP: [Key Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-319](https://cwe.mitre.org/data/definitions/319.html).
