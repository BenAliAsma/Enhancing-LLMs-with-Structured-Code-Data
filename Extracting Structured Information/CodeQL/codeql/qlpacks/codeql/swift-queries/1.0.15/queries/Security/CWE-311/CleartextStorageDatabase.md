# Cleartext storage of sensitive information in a local database
Sensitive information that is stored unencrypted in a database is accessible to an attacker who gains access to that database. For example, the information could be accessed by any process or user in a rooted device, or exposed through another vulnerability.


## Recommendation
Either encrypt the entire database, or ensure that each piece of sensitive information is encrypted before being stored. In general, decrypt sensitive information only at the point where it is necessary for it to be used in cleartext. Avoid storing sensitive information at all if you do not need to keep it.


## Example
The following example shows three cases of storing information using the Core Data library. In the 'BAD' case, the data that is stored is sensitive (a credit card number) and is not encrypted. In the 'GOOD' cases, the data is either not sensitive, or is protected with encryption. When encryption is used, take care to select a secure modern encryption algorithm, and put suitable key management practices into place.


```swift
import CryptoKit

private func encrypt(_ text: String, _ encryptionKey: SymmetricKey) -> String {
	let sealedBox = try! AES.GCM.seal(Data(text.utf8), using: encryptionKey)
	return sealedBox.combined!.base64EncodedString()
}

func storeMyData(databaseObject : NSManagedObject, faveSong : String, creditCardNo : String, encryptionKey: SymmetricKey) {
	// ...

	// GOOD: not sensitive information
	databaseObject.setValue(faveSong, forKey: "myFaveSong")

	// BAD: sensitive information saved in cleartext
	databaseObject.setValue(creditCardNo, forKey: "myCreditCardNo")

	// GOOD: encrypted sensitive information saved
	databaseObject.setValue(encrypt(creditCardNo, encryptionKey), forKey: "myCreditCardNo")

	// ...
}

```

## References
* OWASP Top 10:2021: [A02:2021 � Cryptographic Failures](https://owasp.org/Top10/A02_2021-Cryptographic_Failures/).
* OWASP: [Key Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-312](https://cwe.mitre.org/data/definitions/312.html).
