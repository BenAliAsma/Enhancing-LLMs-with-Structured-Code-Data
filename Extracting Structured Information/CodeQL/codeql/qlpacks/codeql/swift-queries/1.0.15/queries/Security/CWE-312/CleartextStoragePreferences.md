# Cleartext storage of sensitive information in an application preference store
Sensitive information that is stored unencrypted in an application preference store, such as the user defaults database or the iCloud-backed ubiquitous key-value store, is accessible to an attacker who gains access to that data store. For example, the information could be accessed by any process or user in a rooted device, by compromised app extensions, or could be exposed through another vulnerability.


## Recommendation
Either store the data in an encrypted database, or ensure that each piece of sensitive information is encrypted before being stored. In general, decrypt sensitive information only at the point where it is necessary for it to be used in cleartext. Avoid storing sensitive information at all if you do not need to keep it.


## Example
The following example shows three cases of storing information using UserDefaults. In the 'BAD' case, the data that is stored is sensitive (a credit card number) and is not encrypted. In the 'GOOD' cases, the data is either not sensitive, or is protected with encryption. When encryption is used, take care to select a secure modern encryption algorithm, and put suitable key management practices into place.


```swift
import CryptoKit

private func encrypt(_ text: String, _ encryptionKey: SymmetricKey) -> String {
	let sealedBox = try! AES.GCM.seal(Data(text.utf8), using: encryptionKey)
	return sealedBox.combined!.base64EncodedString()
}

func storeMyData(faveSong : String, creditCardNo : String, encryptionKey: SymmetricKey) {
	// ...

	// GOOD: not sensitive information
	UserDefaults.standard.set(faveSong, forKey: "myFaveSong")

	// BAD: sensitive information saved in cleartext
	UserDefaults.standard.set(creditCardNo, forKey: "myCreditCardNo")

	// GOOD: encrypted sensitive information saved
	UserDefaults.standard.set(encrypt(creditCardNo, encryptionKey), forKey: "myCreditCardNo")

	// ...
}

```

## References
* OWASP Top 10:2021: [A02:2021 &mdash; Cryptographic Failures](https://owasp.org/Top10/A02_2021-Cryptographic_Failures/).
* Apple Developer Documentation: [UserDefaults](https://developer.apple.com/documentation/foundation/userdefaults), [NSUbiquitousKeyValueStore](https://developer.apple.com/documentation/foundation/nsubiquitouskeyvaluestore)
* OWASP: [Key Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-312](https://cwe.mitre.org/data/definitions/312.html).
